"""Synthetic SDE plant and thermostat export."""

from pi_nsde_building_thermal.synthetic import SyntheticConfig, generate_synthetic_building


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
