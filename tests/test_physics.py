"""Physics residual helpers."""

import jax.numpy as jnp

from pi_nsde_building_thermal.physics import BuildingParams, dtemp_dt, humidity_ratio, saturation_vapor_pressure_pa


def test_humidity_ratio_typical_range():
    omega = humidity_ratio(jnp.array(20.0), jnp.array(0.4))
    assert 0.004 < float(omega) < 0.01


def test_cooldown_when_indoor_is_warmer_than_outdoor():
    params = BuildingParams(C=9.5, R=3.6, A_s=8.5, beta=120.0)
    dT = dtemp_dt(
        t_in_c=jnp.array(21.0),
        t_out_c=jnp.array(-5.0),
        ghi_w_m2=jnp.array(0.0),
        omega_out=jnp.array(0.002),
        omega_in=jnp.array(0.006),
        q_hvac_kw=jnp.array(0.0),
        q_int_kw=jnp.array(0.0),
        wind_m_s=jnp.array(2.0),
        params=params,
    )
    assert float(dT) < 0.0


def test_known_hvac_raises_derivative():
    params = BuildingParams(C=9.5, R=3.6, A_s=8.5, beta=120.0)
    kwargs = dict(
        t_in_c=jnp.array(19.0),
        t_out_c=jnp.array(-5.0),
        ghi_w_m2=jnp.array(0.0),
        omega_out=jnp.array(0.002),
        omega_in=jnp.array(0.006),
        q_int_kw=jnp.array(0.4),
        wind_m_s=jnp.array(2.0),
        params=params,
    )
    off = dtemp_dt(q_hvac_kw=jnp.array(0.0), **kwargs)
    on = dtemp_dt(q_hvac_kw=jnp.array(9.0), **kwargs)
    assert float(on) > float(off)


def test_saturation_pressure_increases_with_temperature():
    assert float(saturation_vapor_pressure_pa(jnp.array(20.0))) > float(
        saturation_vapor_pressure_pa(jnp.array(0.0))
    )
