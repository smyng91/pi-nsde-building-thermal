"""CSV round-trip and checkpoint save/load."""

import jax
import jax.numpy as jnp
import pandas as pd

from pinn_building.io import (
    estimates_json,
    load_checkpoint,
    load_timeseries_csv,
    save_checkpoint,
    timeseries_from_frame,
    timeseries_to_frame,
)
from pinn_building.model import init_params
from pinn_building.synthetic import SyntheticConfig, generate_synthetic_building


def test_csv_roundtrip_preserves_required_channels(tmp_path):
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    path = tmp_path / "week.csv"
    timeseries_to_frame(data.arrays).to_csv(path, index=False)
    loaded = load_timeseries_csv(path)
    assert loaded.t_in_c.shape == data.arrays.t_in_c.shape
    assert jnp.allclose(loaded.t_in_c, data.arrays.t_in_c, atol=1e-4)
    assert jnp.allclose(loaded.t_out_c, data.arrays.t_out_c, atol=1e-4)
    assert jnp.allclose(loaded.hvac_on_frac, data.arrays.hvac_on_frac, atol=1e-4)


def test_runtime_seconds_become_on_frac():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-15", periods=4, freq="5min"),
            "t_in_c": [20.0, 20.1, 19.8, 19.7],
            "t_out_c": [0.0, -0.2, -0.4, -0.5],
            "hvac_runtime_s": [0.0, 300.0, 150.0, 0.0],
        }
    )
    data = timeseries_from_frame(frame)
    assert float(data.hvac_on_frac[0]) == 0.0
    assert abs(float(data.hvac_on_frac[1]) - 1.0) < 1e-5
    assert abs(float(data.hvac_on_frac[2]) - 0.5) < 1e-5


def test_checkpoint_roundtrip(tmp_path):
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=1))
    params = init_params(jax.random.PRNGKey(0), data.arrays, q_rated_mode="unknown")
    path = tmp_path / "fit.pkl"
    save_checkpoint(path, params, {"q_rated": "unknown", "n_sub": 5})
    loaded, meta = load_checkpoint(path)
    assert meta["q_rated"] == "unknown"
    assert jnp.allclose(loaded.phys.raw_C, params.phys.raw_C)
    assert jnp.allclose(loaded.feat_mean, params.feat_mean)
    body = estimates_json(loaded)
    assert body["C_kwh_per_k"] > 0
    assert body["Q_rated_kw"] > 0
