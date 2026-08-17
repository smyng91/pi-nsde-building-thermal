"""SDE Kalman likelihood and training."""

import jax.numpy as jnp

from pinn_building.model import init_params
from pinn_building.physics import BuildingParams
from pinn_building.sde import SdeNoise, interval_average_kalman, occupancy_mean_kw
from pinn_building.synthetic import SyntheticConfig, generate_synthetic_building
from pinn_building.train import TrainConfig, train_sde


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
        q0=0.7,
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


def test_short_training_runs_and_params_stay_positive():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    result = train_sde(data.arrays, TrainConfig(steps=8, log_every=8, seed=1, q_rated="unknown"), verbose=False)
    assert float(result.estimated.C) > 0
    assert float(result.estimated.R) > 0
    assert float(result.estimated.Q_rated) > 0
    assert result.filter.t_var.shape[0] == data.arrays.t_in_c.shape[0]
