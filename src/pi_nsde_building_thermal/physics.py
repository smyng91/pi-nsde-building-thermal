"""Lumped RC building energy balance.

The estimator identifies an effective indoor node:

    C dT/dt = UA_eff (T_a - T) + A_s I + β (ω_a - ω_i) + Q_hvac + Q_int
    UA_eff  = (1/R) (1 + k_wind v_wind)

equivalent to using R_eff = R / (1 + k_wind v_wind) in (T_a - T) / R_eff.
C is effective thermal capacity, R is envelope resistance (inverse of UA),
A_s is solar aperture, and β converts a humidity-ratio difference into a
latent/infiltration heat flux. Q_hvac is heat into the node: positive for
heating, negative for cooling.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from pi_nsde_building_thermal.constants import (
    ATMOS_PRESSURE_PA,
    MAGNUS_A,
    MAGNUS_B,
    MAGNUS_C,
    WATER_AIR_MASS_RATIO,
    WIND_INFILTRATION_PER_MPS,
)


class BuildingParams(NamedTuple):
    """Effective 1R1C-G parameters in integrator-friendly units."""

    C: float
    """Effective thermal capacity [kWh/K]."""
    R: float
    """Envelope thermal resistance [K/kW] (UA = 1/R)."""
    A_s: float
    """Effective solar aperture [m²]."""
    beta: float
    """Latent / moisture-driven heat coefficient [kW per kg/kg]."""
    Q_rated: float = 6.0
    """Rated HVAC capacity [kW]. Prior/init scale is not plant truth.

    Unknown-Q_rated identification uses ``Q_hvac = Q_rated * u`` with signed
    runtime ``u ∈ [-1, 1]``. The synthetic plant still generates data with
    ``SyntheticConfig.heating_capacity_kw`` as the magnitude.
    """


def saturation_vapor_pressure_pa(temp_c):
    """Magnus-Tetens saturation vapor pressure [Pa]."""
    return MAGNUS_A * jnp.exp(MAGNUS_B * temp_c / (temp_c + MAGNUS_C))


def humidity_ratio(temp_c, rh_frac, pressure_pa: float = ATMOS_PRESSURE_PA):
    """Humidity ratio ω [kg water / kg dry air] from T [°C] and RH in [0, 1]."""
    rh = jnp.clip(rh_frac, 1e-4, 0.999)
    p_v = rh * saturation_vapor_pressure_pa(temp_c)
    p_v = jnp.minimum(p_v, 0.95 * pressure_pa)
    return WATER_AIR_MASS_RATIO * p_v / (pressure_pa - p_v)


def envelope_ua_kw_per_k(
    resistance_k_per_kw,
    wind_m_s,
    wind_k: float = WIND_INFILTRATION_PER_MPS,
):
    """Wind-inflated UA [kW/K]."""
    return (1.0 / resistance_k_per_kw) * (1.0 + wind_k * wind_m_s)


def heat_flux_kw(
    t_in_c,
    t_out_c,
    ghi_w_m2,
    omega_out,
    omega_in,
    q_hvac_kw,
    q_int_kw,
    wind_m_s,
    params: BuildingParams,
    wind_k: float = WIND_INFILTRATION_PER_MPS,
):
    """Net heat into the indoor node [kW]."""
    ua = envelope_ua_kw_per_k(params.R, wind_m_s, wind_k)
    q_envelope = ua * (t_out_c - t_in_c)
    q_solar = params.A_s * ghi_w_m2 / 1000.0
    q_latent = params.beta * (omega_out - omega_in)
    return q_envelope + q_solar + q_latent + q_hvac_kw + q_int_kw


def dtemp_dt(
    t_in_c,
    t_out_c,
    ghi_w_m2,
    omega_out,
    omega_in,
    q_hvac_kw,
    q_int_kw,
    wind_m_s,
    params: BuildingParams,
    wind_k: float = WIND_INFILTRATION_PER_MPS,
):
    """Indoor temperature derivative [K/h] from the energy balance."""
    q = heat_flux_kw(
        t_in_c,
        t_out_c,
        ghi_w_m2,
        omega_out,
        omega_in,
        q_hvac_kw,
        q_int_kw,
        wind_m_s,
        params,
        wind_k,
    )
    return q / params.C
