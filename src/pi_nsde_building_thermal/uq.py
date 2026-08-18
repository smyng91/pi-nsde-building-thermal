"""Practical UQ: Laplace/Hessian on physical parameters from the **train** MAP.

Reported covariance is the inverse Hessian of the train MAP objective with
**sum** of interval NLLs (not mean NLL / N). Physical parameters {C, R, Q_rated
(if unknown), A_s, (β if learned), σ_T, σ_q, σ_y, κ} are joint with Fourier
occupancy coefficients (nuisance); neural remainder/diffusion weights stay at
MAP. Default identifiers freeze β at plant truth, so it is omitted from the
Hessian.

Holdout must not be passed in. Indoor-T Kalman bands are filter state
uncertainty on the series used for UQ (train), not a generalization metric.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pi_nsde_building_thermal.model import ModelParams, PhysRaw, decode_building, decode_noise
from pi_nsde_building_thermal.sde import FilterResult
from pi_nsde_building_thermal.synthetic import Timeseries
from pi_nsde_building_thermal.train import TrainConfig, filter_from_params, map_objective_sum

_EIG_NUMERICAL_FLOOR = 1e-8
_FD_STEP = 2.5e-3

UQ_METHOD = (
    "joint_laplace_central_fd_hessian_of_train_MAP_sum_NLL_"
    "physical_params_joint_with_Fourier_mu_q_neural_weights_at_MAP"
)


class LaplaceUQ(NamedTuple):
    names: tuple[str, ...]
    mean: jnp.ndarray
    sd: jnp.ndarray
    cov_unconstrained: jnp.ndarray
    hessian_unconstrained: jnp.ndarray
    samples: jnp.ndarray
    n_obs: int
    method: str
    eig_min: float
    eig_max: float
    eig_shift: float
    cond: float
    positive_definite: bool


class UncertaintyReport(NamedTuple):
    laplace: LaplaceUQ
    t_mean: jnp.ndarray
    t_sd_state: jnp.ndarray
    t_q05: jnp.ndarray
    t_q95: jnp.ndarray
    q_mean: jnp.ndarray
    q_sd_state: jnp.ndarray


def physics_report_names(q_rated_mode: str, learn_beta: bool = False) -> tuple[str, ...]:
    names: list[str] = ["C", "R"]
    if q_rated_mode == "unknown":
        names.append("Q_rated")
    names.append("A_s")
    if learn_beta:
        names.append("beta")
    names.extend(["sigma_T", "sigma_q", "sigma_y", "kappa"])
    return tuple(names)


def _positive_vector(phys: PhysRaw, q_rated_mode: str, learn_beta: bool) -> jnp.ndarray:
    b = decode_building(phys)
    n = decode_noise(phys)
    parts = [b.C, b.R]
    if q_rated_mode == "unknown":
        parts.append(b.Q_rated)
    parts.append(b.A_s)
    if learn_beta:
        parts.append(b.beta)
    parts.extend([n.sigma_T, n.sigma_q, n.sigma_y, n.kappa])
    return jnp.array(parts)


def _raw_of(val, shift):
    return jnp.log(jnp.expm1(jnp.maximum(val - shift, 1e-3)))


def _raw_from_positive(s: jnp.ndarray, phys: PhysRaw, q_rated_mode: str, learn_beta: bool) -> PhysRaw:
    i = 0
    raw_C = _raw_of(s[i], 0.3)
    i += 1
    raw_R = _raw_of(s[i], 0.3)
    i += 1
    if q_rated_mode == "unknown":
        raw_Q = _raw_of(s[i], 0.3)
        i += 1
    else:
        raw_Q = phys.raw_Q_rated
    raw_As = _raw_of(s[i], 0.2)
    i += 1
    if learn_beta:
        raw_beta = _raw_of(s[i], 1.0)
        i += 1
    else:
        raw_beta = phys.raw_beta
    return PhysRaw(
        raw_C=raw_C,
        raw_R=raw_R,
        raw_Q_rated=raw_Q,
        raw_As=raw_As,
        raw_beta=raw_beta,
        raw_sigma_T=_raw_of(s[i], 0.01),
        raw_sigma_q=_raw_of(s[i + 1], 0.02),
        raw_sigma_y=_raw_of(s[i + 2], 0.02),
        raw_kappa=_raw_of(s[i + 3], 0.15),
        fourier_q=phys.fourier_q,
    )


def _theta(phys: PhysRaw, q_rated_mode: str, learn_beta: bool) -> jnp.ndarray:
    core = [phys.raw_C, phys.raw_R]
    if q_rated_mode == "unknown":
        core.append(phys.raw_Q_rated)
    core.append(phys.raw_As)
    if learn_beta:
        core.append(phys.raw_beta)
    core.extend([phys.raw_sigma_T, phys.raw_sigma_q, phys.raw_sigma_y, phys.raw_kappa])
    return jnp.concatenate([jnp.array(core), phys.fourier_q])


def _splice_theta(theta: jnp.ndarray, phys: PhysRaw, q_rated_mode: str, learn_beta: bool) -> PhysRaw:
    i = 0
    raw_C = theta[i]
    i += 1
    raw_R = theta[i]
    i += 1
    if q_rated_mode == "unknown":
        raw_Q = theta[i]
        i += 1
    else:
        raw_Q = phys.raw_Q_rated
    raw_As = theta[i]
    i += 1
    if learn_beta:
        raw_beta = theta[i]
        i += 1
    else:
        raw_beta = phys.raw_beta
    return phys._replace(
        raw_C=raw_C,
        raw_R=raw_R,
        raw_Q_rated=raw_Q,
        raw_As=raw_As,
        raw_beta=raw_beta,
        raw_sigma_T=theta[i],
        raw_sigma_q=theta[i + 1],
        raw_sigma_y=theta[i + 2],
        raw_kappa=theta[i + 3],
        fourier_q=theta[i + 4 :],
    )


def laplace_on_physics(
    params: ModelParams,
    data: Timeseries,
    train_cfg: TrainConfig,
    n_samples: int = 16,
    seed: int = 0,
    remainder_gate: float = 1.0,
    lambda_id: float | None = None,
) -> LaplaceUQ:
    """Joint Laplace on unconstrained physical params + Fourier μ_q, train series only."""
    mode = train_cfg.q_rated
    learn_beta = bool(train_cfg.learn_beta)
    theta0 = _theta(params.phys, mode, learn_beta)
    n_theta = int(theta0.shape[0])
    n_obs = int(data.t_in_c.shape[0])
    gate = float(remainder_gate)

    def nll_flat(theta):
        p = params._replace(phys=_splice_theta(theta, params.phys, mode, learn_beta))
        loss, _ = map_objective_sum(p, data, train_cfg, remainder_gate=gate, lambda_id=lambda_id)
        return loss

    grad_fn = jax.jit(jax.grad(nll_flat))
    eps = _FD_STEP
    rows = []
    for i in range(n_theta):
        e = jnp.zeros((n_theta,)).at[i].set(eps)
        rows.append((grad_fn(theta0 + e) - grad_fn(theta0 - e)) / (2.0 * eps))
    hess = 0.5 * (jnp.stack(rows) + jnp.stack(rows).T)
    eigs = jnp.linalg.eigvalsh(hess)
    eig_min = float(jnp.min(eigs))
    eig_max = float(jnp.max(eigs))
    positive_definite = eig_min > 0.0
    cond = float(abs(eig_max) / max(abs(eig_min), 1e-12))
    eig_shift = float(max(_EIG_NUMERICAL_FLOOR - eig_min, 0.0))
    hess_pd = hess + eig_shift * jnp.eye(n_theta)
    cov = jnp.linalg.inv(hess_pd)

    jac = jax.jacrev(
        lambda th: _positive_vector(_splice_theta(th, params.phys, mode, learn_beta), mode, learn_beta)
    )(theta0)
    cov_phys = jac @ cov @ jac.T
    sd = jnp.sqrt(jnp.maximum(jnp.diag(cov_phys), 1e-12))
    mean = _positive_vector(params.phys, mode, learn_beta)

    draws = jax.random.multivariate_normal(jax.random.PRNGKey(seed), theta0, cov, shape=(n_samples,))
    samples = jax.vmap(
        lambda th: _positive_vector(_splice_theta(th, params.phys, mode, learn_beta), mode, learn_beta)
    )(draws)
    return LaplaceUQ(
        names=physics_report_names(mode, learn_beta),
        mean=mean,
        sd=sd,
        cov_unconstrained=cov,
        hessian_unconstrained=hess_pd,
        samples=samples,
        n_obs=n_obs,
        method=UQ_METHOD,
        eig_min=eig_min,
        eig_max=eig_max,
        eig_shift=eig_shift,
        cond=cond,
        positive_definite=positive_definite,
    )


def quantify_uncertainty(
    params: ModelParams,
    data: Timeseries,
    filt: FilterResult,
    train_cfg: TrainConfig,
    n_sub: int,
    max_samples: int = 12,
    remainder_gate: float = 1.0,
    lambda_id: float | None = None,
) -> UncertaintyReport:
    """Laplace CIs from ``data`` (must be train). Filter bands are for that same series."""
    laplace = laplace_on_physics(
        params, data, train_cfg, remainder_gate=remainder_gate, lambda_id=lambda_id
    )
    t_sd = jnp.sqrt(filt.t_var)
    q_sd = jnp.sqrt(filt.q_var)
    mode = train_cfg.q_rated
    learn_beta = bool(train_cfg.learn_beta)

    def one(s):
        p = params._replace(phys=_raw_from_positive(s, params.phys, mode, learn_beta))
        return filter_from_params(
            p, data, n_sub, remainder_gate=remainder_gate, q_rated_mode=mode
        ).y_pred

    n_use = min(max_samples, int(laplace.samples.shape[0]))
    if laplace.positive_definite:
        preds = jax.vmap(one)(laplace.samples[:n_use])
        t_q05 = jnp.minimum(jnp.percentile(preds, 5.0, axis=0), filt.t_mean - 2.0 * t_sd)
        t_q95 = jnp.maximum(jnp.percentile(preds, 95.0, axis=0), filt.t_mean + 2.0 * t_sd)
    else:
        t_q05 = filt.t_mean - 2.0 * t_sd
        t_q95 = filt.t_mean + 2.0 * t_sd
    return UncertaintyReport(
        laplace=laplace,
        t_mean=filt.t_mean,
        t_sd_state=t_sd,
        t_q05=t_q05,
        t_q95=t_q95,
        q_mean=filt.q_mean,
        q_sd_state=q_sd,
    )
