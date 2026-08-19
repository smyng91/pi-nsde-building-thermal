"""Synthetic SDE plant and thermostat export."""

import jax.numpy as jnp
import numpy as np

from pi_nsde_building_thermal.synthetic import (
    TRUE_NOISE,
    TRUE_PARAMS,
    SyntheticConfig,
    generate_synthetic_building,
    occupancy_schedule_kw,
)


def test_synthetic_shapes_and_hvac_are_observed():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=1))
    n = data.frame.shape[0]
    assert n == 24 * 12
    for col in (
        "t_out_c",
        "ghi_w_m2",
        "rh_out_frac",
        "wind_m_s",
        "t_in_c",
        "hvac_kw",
        "hvac_on_frac",
        "hvac_runtime_s",
        "heat_setpoint_c",
    ):
        assert col in data.frame.columns
    assert data.arrays.q_hvac_kw.shape == (n,)
    assert float(data.frame["hvac_runtime_s"].max()) > 0
    assert data.true_params.C > 0


def test_indoor_temperature_stays_in_reasonable_band():
    data = generate_synthetic_building(SyntheticConfig(days=2, seed=2))
    t_in = data.frame["t_in_c"].to_numpy()
    assert t_in.min() > 12.0
    assert t_in.max() < 26.0


def test_winter_and_summer_share_building_parameters():
    """Seasonal twins differ in weather and HVAC mode, not in the plant."""
    heat = generate_synthetic_building(SyntheticConfig(days=2, seed=0, hvac_mode="heating"))
    cool = generate_synthetic_building(SyntheticConfig(days=2, seed=0, hvac_mode="cooling"))
    assert heat.true_params == cool.true_params == TRUE_PARAMS
    assert heat.true_noise == cool.true_noise == TRUE_NOISE
    assert heat.config.indoor_rh == cool.config.indoor_rh
    assert heat.config.deadband_k == cool.config.deadband_k
    assert heat.config.q_int_init_kw == cool.config.q_int_init_kw
    t = heat.arrays.t_hours
    assert jnp.allclose(occupancy_schedule_kw(t), occupancy_schedule_kw(cool.arrays.t_hours))
    assert abs(float(heat.arrays.t_out_c.mean()) - float(cool.arrays.t_out_c.mean())) > 10.0
    q_on = float(TRUE_PARAMS.Q_rated)
    for data in (heat, cool):
        q = np.abs(np.asarray(data.arrays.q_hvac_kw))
        assert float(q.max()) <= q_on + 1e-5
        assert float(q.max()) > 0.5 * q_on
