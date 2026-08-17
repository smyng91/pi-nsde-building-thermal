"""Practical UQ: Laplace/Hessian on physical parameters from the **train** MAP.

Reported covariance is the inverse Hessian of the train MAP objective with
**sum** of interval NLLs (not mean NLL / N). Physical parameters {C, R, Q_rated
(if unknown), A_s, β, σ_T, σ_q, σ_y, κ} are joint with Fourier occupancy
coefficients (nuisance); neural remainder/diffusion weights stay at MAP.

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

PHYS_REPORT_NAMES_KNOWN = ("C", "R", "A_s", "beta", "sigma_T", "sigma_q", "sigma_y", "kappa")
PHYS_REPORT_NAMES_UNKNOWN = (
    "C",
    "R",
    "Q_rated",
    "A_s",
    "beta",
    "sigma_T",
    "sigma_q",
    "sigma_y",
    "kappa",
)
UQ_METHOD = (
    "joint_laplace_hessian_of_train_MAP_sum_NLL_"
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


class UncertaintyReport(NamedTuple):
    laplace: LaplaceUQ
    t_mean: jnp.ndarray
    t_sd_state: jnp.ndarray
    t_q05: jnp.ndarray
    t_q95: jnp.ndarray
    q_mean: jnp.ndarray
    q_sd_state: jnp.ndarray


def physics_report_names(q_rated_mode: str) -> tuple[str, ...]:
    if q_rated_mode == "unknown":
        return PHYS_REPORT_NAMES_UNKNOWN
    return PHYS_REPORT_NAMES_KNOWN


def _positive_vector(phys: PhysRaw, q_rated_mode: str) -> jnp.ndarray:
    b = decode_building(phys)
    n = decode_noise(phys)
    if q_rated_mode == "unknown":
        return jnp.array([b.C, b.R, b.Q_rated, b.A_s, b.beta, n.sigma_T, n.sigma_q, n.sigma_y, n.kappa])
    return jnp.array([b.C, b.R, b.A_s, b.beta, n.sigma_T, n.sigma_q, n.sigma_y, n.kappa])


def _raw_from_positive(s: jnp.ndarray, phys: PhysRaw, q_rated_mode: str) -> PhysRaw:
    def raw_of(val, shift):
        return jnp.log(jnp.expm1(jnp.maximum(val - shift, 1e-3)))

    if q_rated_mode == "unknown":
        return PhysRaw(
            raw_C=raw_of(s[0], 0.3),
            raw_R=raw_of(s[1], 0.3),
            raw_Q_rated=raw_of(s[2], 0.3),
            raw_As=raw_of(s[3], 0.2),
            raw_beta=raw_of(s[4], 1.0),
            raw_sigma_T=raw_of(s[5], 0.01),
            raw_sigma_q=raw_of(s[6], 0.02),
            raw_sigma_y=raw_of(s[7], 0.02),
            raw_kappa=raw_of(s[8], 0.15),
            fourier_q=phys.fourier_q,
        )
    return PhysRaw(
        raw_C=raw_of(s[0], 0.3),
        raw_R=raw_of(s[1], 0.3),
        raw_Q_rated=phys.raw_Q_rated,
        raw_As=raw_of(s[2], 0.2),
        raw_beta=raw_of(s[3], 1.0),
        raw_sigma_T=raw_of(s[4], 0.01),
        raw_sigma_q=raw_of(s[5], 0.02),
        raw_sigma_y=raw_of(s[6], 0.02),
        raw_kappa=raw_of(s[7], 0.15),
        fourier_q=phys.fourier_q,
    )


def _theta(phys: PhysRaw, q_rated_mode: str) -> jnp.ndarray:
    if q_rated_mode == "unknown":
        core = jnp.array(
            [
                phys.raw_C,
                phys.raw_R,
                phys.raw_Q_rated,
                phys.raw_As,
                phys.raw_beta,
                phys.raw_sigma_T,
                phys.raw_sigma_q,
                phys.raw_sigma_y,
                phys.raw_kappa,
            ]
        )
    else:
        core = jnp.array(
            [
                phys.raw_C,
                phys.raw_R,
                phys.raw_As,
                phys.raw_beta,
                phys.raw_sigma_T,
                phys.raw_sigma_q,
                phys.raw_sigma_y,
                phys.raw_kappa,
            ]
        )
    return jnp.concatenate([core, phys.fourier_q])


def _splice_theta(theta: jnp.ndarray, phys: PhysRaw, q_rated_mode: str) -> PhysRaw:
    if q_rated_mode == "unknown":
        return phys._replace(
            raw_C=theta[0],
            raw_R=theta[1],
            raw_Q_rated=theta[2],
            raw_As=theta[3],
            raw_beta=theta[4],
            raw_sigma_T=theta[5],
            raw_sigma_q=theta[6],
            raw_sigma_y=theta[7],
            raw_kappa=theta[8],
            fourier_q=theta[9:],
        )
    return phys._replace(
        raw_C=theta[0],
        raw_R=theta[1],
        raw_As=theta[2],
        raw_beta=theta[3],
        raw_sigma_T=theta[4],
        raw_sigma_q=theta[5],
        raw_sigma_y=theta[6],
        raw_kappa=theta[7],
        fourier_q=theta[8:],
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
    theta0 = _theta(params.phys, mode)
    n_theta = int(theta0.shape[0])
    n_obs = int(data.t_in_c.shape[0])
    gate = float(remainder_gate)

    def nll_flat(theta):
        p = params._replace(phys=_splice_theta(theta, params.phys, mode))
        loss, _ = map_objective_sum(p, data, train_cfg, remainder_gate=gate, lambda_id=lambda_id)
        return loss

    grad_fn = jax.jit(jax.grad(nll_flat))
    g0 = grad_fn(theta0)
    eps = 2.5e-3
    rows = []
    for i in range(n_theta):
        e = jnp.zeros((n_theta,)).at[i].set(eps)
        rows.append((grad_fn(theta0 + e) - g0) / eps)
    hess = 0.5 * (jnp.stack(rows) + jnp.stack(rows).T)
    eigs = jnp.linalg.eigvalsh(hess)
    hess_pd = hess + jnp.maximum(1e-3 - jnp.min(eigs), 0.0) * jnp.eye(n_theta)
    cov = jnp.linalg.inv(hess_pd)

    jac = jax.jacrev(lambda th: _positive_vector(_splice_theta(th, params.phys, mode), mode))(theta0)
    cov_phys = jac @ cov @ jac.T
    sd = jnp.sqrt(jnp.maximum(jnp.diag(cov_phys), 1e-12))
    mean = _positive_vector(params.phys, mode)

    draws = jax.random.multivariate_normal(jax.random.PRNGKey(seed), theta0, cov, shape=(n_samples,))
    samples = jax.vmap(lambda th: _positive_vector(_splice_theta(th, params.phys, mode), mode))(draws)
    return LaplaceUQ(
        names=physics_report_names(mode),
        mean=mean,
        sd=sd,
        cov_unconstrained=cov,
        hessian_unconstrained=hess_pd,
        samples=samples,
        n_obs=n_obs,
        method=UQ_METHOD,
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

    def one(s):
        p = params._replace(phys=_raw_from_positive(s, params.phys, mode))
        return filter_from_params(
            p, data, n_sub, remainder_gate=remainder_gate, q_rated_mode=mode
        ).y_pred

    n_use = min(max_samples, int(laplace.samples.shape[0]))
    preds = jax.vmap(one)(laplace.samples[:n_use])
    t_q05 = jnp.minimum(jnp.percentile(preds, 5.0, axis=0), filt.t_mean - 2.0 * t_sd)
    t_q95 = jnp.maximum(jnp.percentile(preds, 95.0, axis=0), filt.t_mean + 2.0 * t_sd)
    return UncertaintyReport(
        laplace=laplace,
        t_mean=filt.t_mean,
        t_sd_state=t_sd,
        t_q05=t_q05,
        t_q95=t_q95,
        q_mean=filt.q_mean,
        q_sd_state=q_sd,
    )
