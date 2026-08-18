"""SDE Kalman likelihood and training."""

import jax.numpy as jnp

from pi_nsde_building_thermal.constants import FILTER_Q0_KW
from pi_nsde_building_thermal.physics import BuildingParams
from pi_nsde_building_thermal.sde import (
    SdeNoise,
    euler_maruyama_matrices,
    exogenous_plus_latent_b,
    interval_average_kalman,
    occupancy_mean_kw,
)
from pi_nsde_building_thermal.synthetic import SyntheticConfig, generate_synthetic_building
from pi_nsde_building_thermal.train import TrainConfig, train_sde


def _kalman_at(building, noise, data, remainder=None, scale=None):
    n = data.t_in_c.shape[0]
    if remainder is None:
        remainder = jnp.zeros((n,))
    if scale is None:
        scale = jnp.ones((n,))
    mu = occupancy_mean_kw(data.t_hours, jnp.array([0.2, 0.15, 0.0, 0.1, 0.0, 0.0, 0.0]))
    dt = float(data.t_hours[1] - data.t_hours[0])
    return interval_average_kalman(
        building,
        noise,
        remainder,
        scale,
        mu,
        data.t_out_c,
        data.ghi_w_m2,
        data.omega_out,
        data.omega_in,
        data.q_hvac_kw,
        data.wind_m_s,
        data.t_in_c,
        n_sub=5,
        dt_sub_h=dt / 5.0,
        t0=data.t_in_c[0],
        q0=FILTER_Q0_KW,
    )


def test_kalman_nll_is_finite():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    filt = _kalman_at(data.true_params, data.true_noise, data.arrays)
    assert jnp.isfinite(filt.nll)
    assert filt.t_mean.shape == data.arrays.t_in_c.shape


def test_nll_prefers_true_capacity_over_far_guess():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0, t_in_noise_k=0.03))
    true_f = _kalman_at(data.true_params, data.true_noise, data.arrays)
    wrong = BuildingParams(C=4.0, R=8.0, A_s=2.0, beta=40.0)
    wrong_n = SdeNoise(sigma_T=0.2, sigma_q=0.4, sigma_y=0.2, kappa=0.4)
    wrong_f = _kalman_at(wrong, wrong_n, data.arrays)
    assert float(true_f.nll) < float(wrong_f.nll)


def test_post_step_accumulator_matches_explicit_em_mean():
    """Identifier interval mean is the mean of post-step T (F3 accumulator).

    The plant exporter averages incoming 1-minute states (pre-update). That is
    a one-substep index offset, not the same discrete mean.
    """
    dt = 1.0 / 60.0
    n_sub = 5
    ua = 0.28
    C = 9.5
    kappa = 1.25
    t0, q0 = 20.0, 0.6
    f2, f3, qd = euler_maruyama_matrices(ua, C, kappa, dt, sigma_T=0.0, sigma_q=0.0)
    del qd
    g = exogenous_plus_latent_b(
        t_out_c=jnp.array(-4.0),
        ghi_w_m2=jnp.array(80.0),
        omega_out=jnp.array(0.002),
        omega_in=jnp.array(0.006),
        q_hvac_kw=jnp.array(9.0),
        remainder_kw=jnp.array(0.0),
        mu_q_kw=jnp.array(0.7),
        ua_kw_per_k=jnp.array(ua),
        params=BuildingParams(C=C, R=3.6, A_s=8.5, beta=120.0, Q_rated=9.0),
        capacity_kwh_per_k=C,
        kappa=kappa,
        dt_h=dt,
    )
    x = jnp.array([t0, q0])
    temps = []
    for _ in range(n_sub):
        x = f2 @ x + g[:2]
        temps.append(x[0])
    explicit = float(jnp.mean(jnp.stack(temps)))

    m3 = jnp.array([t0, q0, 0.0])
    for _ in range(n_sub):
        m3 = f3 @ m3 + g
    accum = float(m3[2] / n_sub)
    assert abs(accum - explicit) < 1e-5
    assert abs(float(m3[0]) - float(temps[-1])) < 1e-5


def test_short_training_runs_and_params_stay_positive():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    result = train_sde(data.arrays, TrainConfig(steps=8, log_every=8, seed=1, q_rated="unknown"), verbose=False)
    assert float(result.estimated.C) > 0
    assert float(result.estimated.R) > 0
    assert float(result.estimated.Q_rated) > 0
    assert result.filter.t_var.shape[0] == data.arrays.t_in_c.shape[0]
