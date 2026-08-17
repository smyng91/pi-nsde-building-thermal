"""Physics-informed RC SDE: drift, Euler–Maruyama, interval-average Kalman likelihood.

State
-----
T      indoor temperature [°C]
Q_int  latent internal-gain / occupancy process [kW]

    dT     = (1/C) [ UA_eff (T_a - T) + A_s I + β Δω + Q_hvac + Q_int + r_θ ] dt
             + σ_T(u) dW_T
    dQ_int = κ (μ(t) - Q_int) dt + σ_q dW_q

Q_hvac is an exogenous input: either metered kW (known-Q_rated protocol) or
``Q_rated * u_on`` with observed interval runtime ``u_on`` (unknown capacity).
HVAC on/off is never a latent switching mode. Observations are interval
averages of T, not Dirac samples at the endpoints.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pinn_building.physics import BuildingParams, dtemp_dt, envelope_ua_kw_per_k


class SdeNoise(NamedTuple):
    sigma_T: float
    """Temperature diffusion [K / sqrt(h)]."""
    sigma_q: float
    """Internal-gain diffusion [kW / sqrt(h)]."""
    sigma_y: float
    """Measurement noise on the interval-average temperature [K]."""
    kappa: float
    """OU mean-reversion rate for Q_int [1/h]."""


class FilterResult(NamedTuple):
    nll: jnp.ndarray
    t_mean: jnp.ndarray
    t_var: jnp.ndarray
    q_mean: jnp.ndarray
    q_var: jnp.ndarray
    y_pred: jnp.ndarray
    innov_var: jnp.ndarray


def occupancy_mean_kw(t_hours, fourier_raw: jnp.ndarray) -> jnp.ndarray:
    """Positive daily-periodic mean occupancy/internal gain [kW]."""
    t = jnp.atleast_1d(t_hours)
    q = jnp.full_like(t, fourier_raw[0])
    n_harm = (fourier_raw.shape[0] - 1) // 2
    for k in range(1, n_harm + 1):
        ang = 2.0 * jnp.pi * k * t / 24.0
        q = q + fourier_raw[2 * k - 1] * jnp.sin(ang) + fourier_raw[2 * k] * jnp.cos(ang)
    return jax.nn.softplus(q)


def euler_maruyama_matrices(
    ua_kw_per_k,
    capacity_kwh_per_k,
    kappa,
    dt_h,
    sigma_T,
    sigma_q,
):
    """Affine EM maps for x=[T, Q_int] and running sum S of T.

    x' = F_2 x + g_2,  S' = S + T.
    """
    a_t = -ua_kw_per_k / capacity_kwh_per_k
    a_q = 1.0 / capacity_kwh_per_k
    f2 = jnp.array(
        [
            [1.0 + dt_h * a_t, dt_h * a_q],
            [0.0, 1.0 - dt_h * kappa],
        ]
    )
    f3 = jnp.array(
        [
            [1.0 + dt_h * a_t, dt_h * a_q, 0.0],
            [0.0, 1.0 - dt_h * kappa, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    qd = jnp.diag(jnp.array([dt_h * sigma_T**2, dt_h * sigma_q**2, 0.0]))
    return f2, f3, qd


def exogenous_plus_latent_b(
    t_out_c,
    ghi_w_m2,
    omega_out,
    omega_in,
    q_hvac_kw,
    remainder_kw,
    mu_q_kw,
    ua_kw_per_k,
    params: BuildingParams,
    capacity_kwh_per_k,
    kappa,
    dt_h,
):
    """Affine intercept g for [T, Q, S] after one EM step (S intercept 0)."""
    q_sol = params.A_s * ghi_w_m2 / 1000.0
    q_lat = params.beta * (omega_out - omega_in)
    b_t = (ua_kw_per_k * t_out_c + q_sol + q_lat + q_hvac_kw + remainder_kw) / capacity_kwh_per_k
    b_q = kappa * mu_q_kw
    return jnp.array([dt_h * b_t, dt_h * b_q, 0.0])


def interval_average_kalman(
    params: BuildingParams,
    noise: SdeNoise,
    remainder_kw: jnp.ndarray,
    sigma_T_scale: jnp.ndarray,
    mu_q_kw: jnp.ndarray,
    t_out_c: jnp.ndarray,
    ghi_w_m2: jnp.ndarray,
    omega_out: jnp.ndarray,
    omega_in: jnp.ndarray,
    q_hvac_kw: jnp.ndarray,
    wind_m_s: jnp.ndarray,
    t_in_avg_c: jnp.ndarray,
    n_sub: int,
    dt_sub_h: float,
    t0: float,
    q0: float,
) -> FilterResult:
    """Kalman filter for the linear SDE with interval-average T observations.

    Each reporting interval holds weather and Q_hvac constant (ZOH on the
    thermostat export), runs ``n_sub`` Euler–Maruyama steps, and observes
    the mean of the substep indoor temperatures.
    """
    n = t_in_avg_c.shape[0]
    ua = envelope_ua_kw_per_k(params.R, wind_m_s)
    sigma_T = noise.sigma_T * sigma_T_scale
    sigma_q = noise.sigma_q
    sigma_y2 = noise.sigma_y**2

    m0 = jnp.array([t0, q0])
    p0 = jnp.diag(jnp.array([0.6**2, 1.0**2]))

    def interval(carry, k):
        m, p = carry
        f2, f3, qd = euler_maruyama_matrices(
            ua[k], params.C, noise.kappa, dt_sub_h, sigma_T[k], sigma_q
        )
        g = exogenous_plus_latent_b(
            t_out_c[k],
            ghi_w_m2[k],
            omega_out[k],
            omega_in[k],
            q_hvac_kw[k],
            remainder_kw[k],
            mu_q_kw[k],
            ua[k],
            params,
            params.C,
            noise.kappa,
            dt_sub_h,
        )
        m3 = jnp.array([m[0], m[1], 0.0])
        p3 = jnp.zeros((3, 3))
        p3 = p3.at[:2, :2].set(p)

        def sub(carry, _):
            mm, pp = carry
            mm = f3 @ mm + g
            pp = f3 @ pp @ f3.T + qd
            return (mm, pp), None

        (m3, p3), _ = jax.lax.scan(sub, (m3, p3), None, length=n_sub)
        h = jnp.array([0.0, 0.0, 1.0 / n_sub])
        y_pred = h @ m3
        s = h @ p3 @ h + sigma_y2
        k_gain = (p3 @ h) / s
        inn = t_in_avg_c[k] - y_pred
        m3 = m3 + k_gain * inn
        ikh = jnp.eye(3) - jnp.outer(k_gain, h)
        p3 = ikh @ p3 @ ikh.T + sigma_y2 * jnp.outer(k_gain, k_gain)
        nll = 0.5 * (jnp.log(2.0 * jnp.pi * s) + inn**2 / s)
        m2 = m3[:2]
        p2 = 0.5 * (p3[:2, :2] + p3[:2, :2].T)
        return (m2, p2), (nll, m2[0], p2[0, 0], m2[1], p2[1, 1], y_pred, s)

    (_, _), seq = jax.lax.scan(interval, (m0, p0), jnp.arange(n))
    nlls, t_mean, t_var, q_mean, q_var, y_pred, innov_var = seq
    return FilterResult(
        nll=jnp.sum(nlls),
        t_mean=t_mean,
        t_var=jnp.maximum(t_var, 1e-8),
        q_mean=q_mean,
        q_var=jnp.maximum(q_var, 1e-8),
        y_pred=y_pred,
        innov_var=jnp.maximum(innov_var, 1e-8),
    )


class OpenLoopResult(NamedTuple):
    """Mean EM rollout of interval-average T. No Kalman update — holdout T is not used."""

    y_pred: jnp.ndarray
    t_mean: jnp.ndarray
    q_mean: jnp.ndarray


def interval_average_open_loop(
    params: BuildingParams,
    remainder_kw: jnp.ndarray,
    mu_q_kw: jnp.ndarray,
    t_out_c: jnp.ndarray,
    ghi_w_m2: jnp.ndarray,
    omega_out: jnp.ndarray,
    omega_in: jnp.ndarray,
    q_hvac_kw: jnp.ndarray,
    wind_m_s: jnp.ndarray,
    n_sub: int,
    dt_sub_h: float,
    t0: float,
    q0: float,
    kappa: float,
) -> OpenLoopResult:
    """Physics (+ remainder, μ_q) open-loop interval-average T.

    Exogenous inputs only: weather and known HVAC. Indoor T is not an input.
    """
    n = t_out_c.shape[0]
    ua = envelope_ua_kw_per_k(params.R, wind_m_s)
    m0 = jnp.array([t0, q0])

    def interval(m, k):
        f2, f3, qd = euler_maruyama_matrices(
            ua[k], params.C, kappa, dt_sub_h, sigma_T=0.0, sigma_q=0.0
        )
        del f2, qd
        g = exogenous_plus_latent_b(
            t_out_c[k],
            ghi_w_m2[k],
            omega_out[k],
            omega_in[k],
            q_hvac_kw[k],
            remainder_kw[k],
            mu_q_kw[k],
            ua[k],
            params,
            params.C,
            kappa,
            dt_sub_h,
        )
        m3 = jnp.array([m[0], m[1], 0.0])

        def sub(mm, _):
            return f3 @ mm + g, None

        m3, _ = jax.lax.scan(sub, m3, None, length=n_sub)
        y_pred = m3[2] / n_sub
        m2 = m3[:2]
        return m2, (y_pred, m2[0], m2[1])

    _, seq = jax.lax.scan(interval, m0, jnp.arange(n))
    y_pred, t_mean, q_mean = seq
    return OpenLoopResult(y_pred=y_pred, t_mean=t_mean, q_mean=q_mean)


def simulate_em_step(
    t_in_c,
    q_int_kw,
    t_out_c,
    ghi_w_m2,
    omega_out,
    omega_in,
    q_hvac_kw,
    wind_m_s,
    mu_q_kw,
    params: BuildingParams,
    noise: SdeNoise,
    dt_h: float,
    dW_t: float,
    dW_q: float,
):
    """One Euler–Maruyama step of the true (or fitted) plant. HVAC is an input."""
    dT = dtemp_dt(
        t_in_c,
        t_out_c,
        ghi_w_m2,
        omega_out,
        omega_in,
        q_hvac_kw,
        q_int_kw,
        wind_m_s,
        params,
    )
    t_next = t_in_c + dt_h * dT + noise.sigma_T * jnp.sqrt(dt_h) * dW_t
    q_next = q_int_kw + dt_h * noise.kappa * (mu_q_kw - q_int_kw) + noise.sigma_q * jnp.sqrt(dt_h) * dW_q
    return t_next, q_next
