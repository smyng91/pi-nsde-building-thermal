"""Learnable physical {C, R, Q_rated, …} plus a constrained neural remainder/diffusion.

Features are contemporaneous (available at interval k): weather, HVAC on/off
(or metered HVAC power in the optimistic metered-Q_hvac protocol), and
clock-time Fourier terms. Indoor T and hidden Q_int are never remainder inputs.
True delivered HVAC kW must not enter unknown-Q_rated features.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from pi_nsde_building_thermal.physics import DEFAULT_PRIOR, BuildingParams
from pi_nsde_building_thermal.sde import SdeNoise
from pi_nsde_building_thermal.synthetic import Timeseries

QRatedMode = Literal["known", "unknown"]
Q_RATED_MODES = ("known", "unknown")
HvacMode = Literal["auto", "heating", "cooling"]
HVAC_MODES = ("auto", "heating", "cooling")

# Remainder features: exogenous + clock only. No T_in, no Q_int.
# HVAC slot is swapped in exogenous_features: on_frac (unknown) vs kW (known).
CAUSAL_FEATURE_NAMES = (
    "t_out_c",
    "ghi_kw_m2",
    "rh_out",
    "wind_m_s",
    "hvac",
    "sin_hour",
    "cos_hour",
    "sin_2hour",
    "cos_2hour",
)


def normalize_q_rated_mode(mode: str) -> QRatedMode:
    m = str(mode).lower().strip()
    if m not in Q_RATED_MODES:
        raise ValueError(f"q_rated must be 'known' or 'unknown', got {mode!r}")
    return m  # type: ignore[return-value]


def normalize_hvac_mode(mode: str) -> HvacMode:
    m = str(mode).lower().strip()
    if m in {"heat", "heating"}:
        return "heating"
    if m in {"cool", "cooling"}:
        return "cooling"
    if m in {"auto", "signed", "reverse", "heatpump", "heat-pump"}:
        return "auto"
    raise ValueError(f"hvac_mode must be 'auto', 'heating', or 'cooling', got {mode!r}")


def signed_runtime(u, hvac_mode: str = "auto"):
    """Map observed runtime to signed heat-into-node fraction in [-1, 1].

    Positive is heating; negative is cooling. ``Q_rated`` stays positive.
    Unsigned [0, 1] heating data is unchanged. Unsigned cooling runtime is
    negated when ``hvac_mode='cooling'``. Values already below 0 are treated
    as signed (mixed / reverse-cycle) and are left alone.
    """
    mode = normalize_hvac_mode(hvac_mode)
    u = jnp.clip(jnp.asarray(u), -1.0, 1.0)
    already_signed = jnp.min(u) < -1e-8
    unsigned = jnp.clip(u, 0.0, 1.0)
    if mode == "cooling":
        return jnp.where(already_signed, u, -unsigned)
    return u


def align_kw_to_runtime(q_hvac_kw, u_signed):
    """If delivered kW is stored as a magnitude, give it the sign of runtime."""
    q = jnp.asarray(q_hvac_kw)
    u = jnp.asarray(u_signed)
    unsigned_kw = jnp.min(q) >= -1e-8
    has_cooling = jnp.min(u) < -1e-8
    aligned = jnp.sign(u) * jnp.abs(q)
    return jnp.where(unsigned_kw & has_cooling, aligned, q)


def canonicalize_hvac(data: Timeseries, hvac_mode: str = "auto") -> Timeseries:
    """Return a copy whose HVAC channels are signed heat-into-the-node.

    Idempotent: cooling runtime that is already negative is not flipped twice.
    """
    u = signed_runtime(data.hvac_on_frac, hvac_mode)
    q = align_kw_to_runtime(data.q_hvac_kw, u)
    return data._replace(hvac_on_frac=u, q_hvac_kw=q)


def inv_softplus(y: float) -> float:
    y = max(float(y), 1e-4)
    return math.log(math.expm1(y))


class PhysRaw(NamedTuple):
    raw_C: jnp.ndarray
    raw_R: jnp.ndarray
    raw_Q_rated: jnp.ndarray
    raw_As: jnp.ndarray
    raw_beta: jnp.ndarray
    raw_sigma_T: jnp.ndarray
    raw_sigma_q: jnp.ndarray
    raw_sigma_y: jnp.ndarray
    raw_kappa: jnp.ndarray
    fourier_q: jnp.ndarray


class ModelParams(NamedTuple):
    phys: PhysRaw
    remainder_net: list
    sigma_net: list
    feat_mean: jnp.ndarray
    feat_std: jnp.ndarray


def _glorot(key, n_in: int, n_out: int):
    lim = jnp.sqrt(6.0 / (n_in + n_out))
    return random.uniform(key, (n_in, n_out), minval=-lim, maxval=lim), jnp.zeros((n_out,))


def _mlp(net, x):
    for w, b in net[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = net[-1]
    return jnp.squeeze(x @ w + b, axis=-1)


def _zero_last_layer(net: list) -> list:
    """Zero last-layer weights and biases so Stage A remainder/diffusion start at 0."""
    w, b = net[-1]
    return list(net[:-1]) + [(jnp.zeros_like(w), jnp.zeros_like(b))]


def hvac_feature(data: Timeseries, q_rated_mode: str = "unknown") -> jnp.ndarray:
    """Observed HVAC channel: signed runtime, or metered HVAC power in known mode."""
    if q_rated_mode == "unknown":
        return data.hvac_on_frac
    return data.q_hvac_kw


def observed_hvac_kw(params: "ModelParams", data: Timeseries, q_rated_mode: str = "unknown") -> jnp.ndarray:
    """Delivered HVAC power into the indoor node [kW] passed to the SDE.

    Unknown mode: ``Q_hvac = Q_rated * u`` with a learnable positive constant
    rated capacity and signed runtime ``u ∈ [-1, 1]``. Never reads
    ``data.q_hvac_kw``. Known mode: metered plant HVAC power ``q_hvac_kw``
    (negative when cooling). Call ``canonicalize_hvac`` first so unsigned
    cooling runtime is negated.
    """
    if q_rated_mode == "unknown":
        return decode_building(params.phys).Q_rated * data.hvac_on_frac
    return data.q_hvac_kw


def exogenous_features(data: Timeseries, q_rated_mode: str = "unknown") -> jnp.ndarray:
    """Causal features at interval k (ZOH weather/HVAC + clock). No T_in or Q_int."""
    t = data.t_hours
    hour_ang = 2.0 * jnp.pi * t / 24.0
    return jnp.stack(
        [
            data.t_out_c,
            data.ghi_w_m2 / 1000.0,
            data.rh_out,
            data.wind_m_s,
            hvac_feature(data, q_rated_mode),
            jnp.sin(hour_ang),
            jnp.cos(hour_ang),
            jnp.sin(2.0 * hour_ang),
            jnp.cos(2.0 * hour_ang),
        ],
        axis=-1,
    )


def decode_building(phys: PhysRaw) -> BuildingParams:
    return BuildingParams(
        C=jax.nn.softplus(phys.raw_C) + 0.3,
        R=jax.nn.softplus(phys.raw_R) + 0.3,
        A_s=jax.nn.softplus(phys.raw_As) + 0.2,
        beta=jax.nn.softplus(phys.raw_beta) + 1.0,
        Q_rated=jax.nn.softplus(phys.raw_Q_rated) + 0.3,
    )


def decode_noise(phys: PhysRaw) -> SdeNoise:
    return SdeNoise(
        sigma_T=jax.nn.softplus(phys.raw_sigma_T) + 0.01,
        sigma_q=jax.nn.softplus(phys.raw_sigma_q) + 0.02,
        sigma_y=jax.nn.softplus(phys.raw_sigma_y) + 0.02,
        kappa=jax.nn.softplus(phys.raw_kappa) + 0.15,
    )


def init_params(
    key,
    data: Timeseries,
    prior: BuildingParams | None = None,
    q_rated_mode: str = "unknown",
) -> ModelParams:
    """Feature mean/std must be computed on **train** only (caller passes the train slice)."""
    prior = prior or DEFAULT_PRIOR
    mode = normalize_q_rated_mode(q_rated_mode)
    feat = exogenous_features(data, q_rated_mode=mode)
    feat_mean = jnp.mean(feat, axis=0)
    feat_std = jnp.std(feat, axis=0) + 1e-3
    k1, k2 = random.split(key)
    in_dim = feat.shape[-1]

    def make_net(k, hidden=(16, 16)):
        dims = (in_dim, *hidden, 1)
        keys = random.split(k, len(dims) - 1)
        return [_glorot(kk, a, b) for kk, a, b in zip(keys, dims[:-1], dims[1:])]

    # Q_rated init is the prior (6 kW), not plant truth (9 kW).
    phys = PhysRaw(
        raw_C=jnp.array(inv_softplus(prior.C - 0.3)),
        raw_R=jnp.array(inv_softplus(prior.R - 0.3)),
        raw_Q_rated=jnp.array(inv_softplus(prior.Q_rated - 0.3)),
        raw_As=jnp.array(inv_softplus(prior.A_s - 0.2)),
        raw_beta=jnp.array(inv_softplus(prior.beta - 1.0)),
        raw_sigma_T=jnp.array(inv_softplus(0.10)),
        raw_sigma_q=jnp.array(inv_softplus(0.18)),
        raw_sigma_y=jnp.array(inv_softplus(0.10)),
        raw_kappa=jnp.array(inv_softplus(0.9)),
        fourier_q=jnp.array([inv_softplus(0.7), 0.2, -0.05, 0.12, 0.04, 0.02, -0.02]),
    )
    return ModelParams(
        phys=phys,
        remainder_net=_zero_last_layer(make_net(k1)),
        sigma_net=_zero_last_layer(make_net(k2, hidden=(8, 8))),
        feat_mean=feat_mean,
        feat_std=feat_std,
    )


def remainder_and_sigma_scale(
    params: ModelParams,
    data: Timeseries,
    remainder_gate: float = 1.0,
    sigma_gate: float = 1.0,
    q_rated_mode: str = "unknown",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Neural remainder/diffusion. ``remainder_gate=0`` freezes the correction at 0."""
    z = (exogenous_features(data, q_rated_mode=q_rated_mode) - params.feat_mean) / params.feat_std
    r = remainder_gate * 0.18 * jnp.tanh(_mlp(params.remainder_net, z))
    log_s = sigma_gate * 0.35 * jnp.clip(_mlp(params.sigma_net, z), -2.0, 2.0)
    return r, jnp.exp(0.5 * log_s)


def pearson_r2(a, b) -> jnp.ndarray:
    """Ridge-stabilized squared Pearson correlation (population moments)."""
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    return (jnp.mean(a * b) ** 2) / ((jnp.mean(a**2) + 1e-6) * (jnp.mean(b**2) + 1e-6))


def remainder_diagnostics(
    params: ModelParams,
    data: Timeseries,
    remainder_gate: float = 1.0,
    q_rated_mode: str = "unknown",
) -> dict[str, float]:
    """Train-set remainder and σ_T(φ) summaries. Not a holdout metric."""
    r, scale = remainder_and_sigma_scale(
        params,
        data,
        remainder_gate=remainder_gate,
        sigma_gate=remainder_gate,
        q_rated_mode=q_rated_mode,
    )
    sigma = decode_noise(params.phys).sigma_T * scale
    dt = data.t_out_c - data.t_in_c
    ghi = data.ghi_w_m2 / 1000.0
    qh = hvac_feature(data, q_rated_mode)
    return {
        "rms": float(jnp.sqrt(jnp.mean(r**2))),
        "max_abs": float(jnp.max(jnp.abs(r))),
        "rho2_dT": float(pearson_r2(r, dt)),
        "rho2_I": float(pearson_r2(r, ghi)),
        "rho2_hvac": float(pearson_r2(r, qh)),
        "sigma_T_mean": float(jnp.mean(sigma)),
        "sigma_T_min": float(jnp.min(sigma)),
        "sigma_T_max": float(jnp.max(sigma)),
    }


def identifiability_penalty(
    remainder_kw: jnp.ndarray,
    data: Timeseries,
    q_rated_mode: str = "unknown",
) -> jnp.ndarray:
    """Stop the neural remainder from absorbing UA, solar aperture, or 1/C.

    HVAC runtime is observed (signed: heating positive, cooling negative); a remainder
    correlated with that channel (or metered HVAC power) would bias C and
    Q_rated. Indoor T enters only this train-set regularizer, not the remainder features.
    """
    r = remainder_kw
    dt = data.t_out_c - data.t_in_c
    ghi = data.ghi_w_m2 / 1000.0
    qh = hvac_feature(data, q_rated_mode)
    return jnp.mean(r**2) + pearson_r2(r, dt) + pearson_r2(r, ghi) + pearson_r2(r, qh)


def weak_prior(phys: PhysRaw, prior: BuildingParams) -> jnp.ndarray:
    b = decode_building(phys)
    return (
        0.5 * ((jnp.log(b.C) - jnp.log(prior.C)) / 0.7) ** 2
        + 0.5 * ((jnp.log(b.R) - jnp.log(prior.R)) / 0.7) ** 2
        + 0.5 * ((jnp.log(b.Q_rated) - jnp.log(prior.Q_rated)) / 0.7) ** 2
        + 0.15 * ((jnp.log(b.A_s) - jnp.log(prior.A_s)) / 0.8) ** 2
    )


def occupancy_regularizer(phys: PhysRaw) -> jnp.ndarray:
    """L2 / log-prior on Fourier μ_q and occupancy noise so Q_int cannot freely alias UA and 1/C.

    Q_int stays a latent OU; this does not clamp it to the hidden series.
    """
    fq = phys.fourier_q
    noise = decode_noise(phys)
    dc_target = inv_softplus(0.7)
    return (
        0.5 * ((fq[0] - dc_target) / 0.40) ** 2
        + 0.5 * jnp.sum((fq[1:] / 0.22) ** 2)
        + 0.5 * ((jnp.log(noise.sigma_q) - jnp.log(0.16)) / 0.45) ** 2
        + 0.15 * ((jnp.log(noise.kappa) - jnp.log(1.0)) / 0.55) ** 2
    )
