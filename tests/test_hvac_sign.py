"""Heating, cooling, and reverse-cycle signed HVAC runtime."""

import jax
import jax.numpy as jnp
import pandas as pd

from pi_nsde_building_thermal.io import timeseries_from_frame
from pi_nsde_building_thermal.model import (
    canonicalize_hvac,
    decode_building,
    exogenous_features,
    init_params,
    observed_hvac_kw,
    signed_runtime,
)
from pi_nsde_building_thermal.physics import BuildingParams, dtemp_dt
from pi_nsde_building_thermal.synthetic import SyntheticConfig, generate_synthetic_building
from pi_nsde_building_thermal.train import TrainConfig, filter_from_params, train_sde


def test_heating_runtime_stays_nonnegative():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    u = canonicalize_hvac(data.arrays, "auto").hvac_on_frac
    assert float(jnp.min(u)) >= -1e-6
    assert float(jnp.max(u)) <= 1.0 + 1e-6
    assert float(jnp.max(u)) > 0.05
    assert float(jnp.min(data.arrays.q_hvac_kw)) >= -1e-5


def test_cooling_twin_uses_negative_runtime_and_kw():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=3, hvac_mode="cooling"))
    assert float(jnp.min(data.arrays.hvac_on_frac)) < -0.05
    assert float(jnp.max(data.arrays.hvac_on_frac)) <= 1e-6
    assert float(jnp.min(data.arrays.q_hvac_kw)) < -0.5
    assert float(jnp.max(data.arrays.q_hvac_kw)) <= 1e-5
    t_in = data.frame["t_in_c"].to_numpy()
    assert t_in.min() > 16.0
    assert t_in.max() < 34.0


def test_cooling_lowers_indoor_temperature_derivative():
    params = BuildingParams(C=9.5, R=3.6, A_s=8.5, beta=120.0)
    kwargs = dict(
        t_in_c=jnp.array(25.0),
        t_out_c=jnp.array(32.0),
        ghi_w_m2=jnp.array(0.0),
        omega_out=jnp.array(0.012),
        omega_in=jnp.array(0.008),
        q_int_kw=jnp.array(0.4),
        wind_m_s=jnp.array(2.0),
        params=params,
    )
    off = dtemp_dt(q_hvac_kw=jnp.array(0.0), **kwargs)
    cool = dtemp_dt(q_hvac_kw=jnp.array(-9.0), **kwargs)
    assert float(cool) < float(off)


def test_unsigned_cooling_csv_needs_hvac_mode_cooling():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-07-15", periods=4, freq="5min"),
            "t_in_c": [25.0, 25.2, 24.8, 24.7],
            "t_out_c": [32.0, 32.2, 32.4, 32.5],
            "hvac_on_frac": [0.0, 1.0, 0.5, 0.0],
        }
    )
    heating_default = timeseries_from_frame(frame, hvac_mode="auto")
    assert abs(float(heating_default.hvac_on_frac[1]) - 1.0) < 1e-5
    cooling = timeseries_from_frame(frame, hvac_mode="cooling")
    assert abs(float(cooling.hvac_on_frac[1]) + 1.0) < 1e-5
    assert abs(float(cooling.hvac_on_frac[2]) + 0.5) < 1e-5


def test_cooling_on_column_is_negative_without_mode_flag():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-07-15", periods=4, freq="5min"),
            "t_in_c": [25.0, 25.2, 24.8, 24.7],
            "t_out_c": [32.0, 32.2, 32.4, 32.5],
            "cooling_on": [0.0, 1.0, 0.5, 0.0],
        }
    )
    data = timeseries_from_frame(frame, hvac_mode="auto")
    assert abs(float(data.hvac_on_frac[1]) + 1.0) < 1e-5
    assert abs(float(data.hvac_on_frac[2]) + 0.5) < 1e-5


def test_mixed_heat_and_cool_columns_are_signed():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-04-01", periods=4, freq="5min"),
            "t_in_c": [20.0, 20.1, 21.0, 20.8],
            "t_out_c": [10.0, 12.0, 22.0, 21.0],
            "heating_on": [1.0, 0.0, 0.0, 0.0],
            "cooling_on": [0.0, 0.0, 1.0, 0.4],
        }
    )
    data = timeseries_from_frame(frame, hvac_mode="auto")
    assert abs(float(data.hvac_on_frac[0]) - 1.0) < 1e-5
    assert abs(float(data.hvac_on_frac[1])) < 1e-5
    assert abs(float(data.hvac_on_frac[2]) + 1.0) < 1e-5
    assert abs(float(data.hvac_on_frac[3]) + 0.4) < 1e-5


def test_unknown_mode_cooling_q_hvac_is_negative():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=4, hvac_mode="cooling")).arrays
    params = init_params(jax.random.PRNGKey(1), data, q_rated_mode="unknown")
    q = observed_hvac_kw(params, canonicalize_hvac(data, "auto"), "unknown")
    q_hat = float(decode_building(params.phys).Q_rated)
    assert q_hat > 0
    assert float(jnp.min(q)) < -0.05
    assert float(jnp.max(q)) <= 1e-5
    feat = exogenous_features(canonicalize_hvac(data, "auto"), q_rated_mode="unknown")
    assert jnp.allclose(feat[:, 4], canonicalize_hvac(data, "auto").hvac_on_frac)


def test_known_mode_aligns_positive_cooling_kw_to_signed_runtime():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-07-15", periods=3, freq="5min"),
            "t_in_c": [25.0, 25.1, 24.9],
            "t_out_c": [33.0, 33.1, 33.2],
            "cooling_on": [0.0, 1.0, 0.5],
            "hvac_kw": [0.0, 9.0, 4.5],
        }
    )
    data = timeseries_from_frame(frame, hvac_mode="auto")
    assert abs(float(data.q_hvac_kw[1]) + 9.0) < 1e-4
    assert abs(float(data.q_hvac_kw[2]) + 4.5) < 1e-4
    params = init_params(jax.random.PRNGKey(0), data, q_rated_mode="known")
    q = observed_hvac_kw(params, data, "known")
    assert abs(float(q[1]) + 9.0) < 1e-4


def test_canonicalize_cooling_is_idempotent():
    u = jnp.array([0.0, 0.8, 0.2])
    once = signed_runtime(u, "cooling")
    twice = signed_runtime(once, "cooling")
    assert jnp.allclose(once, twice)
    assert abs(float(once[1]) + 0.8) < 1e-6


def test_cooling_unknown_filter_runs():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=5, hvac_mode="cooling")).arrays
    data = canonicalize_hvac(data, "auto")
    result = train_sde(data, TrainConfig(steps=6, log_every=6, seed=1, q_rated="unknown"), verbose=False)
    assert result.hvac_mode == "auto"
    q = observed_hvac_kw(result.params, data, "unknown")
    assert float(jnp.min(q)) < 0
    filt = filter_from_params(result.params, data, 5, remainder_gate=0.0, q_rated_mode="unknown")
    assert jnp.isfinite(filt.nll)
