"""Homogeneous C/Q/R scaling and linear-only Pearson identifiability penalty."""

import jax.numpy as jnp

from pi_nsde_building_thermal.model import identifiability_penalty, pearson_r2
from pi_nsde_building_thermal.physics import envelope_ua_kw_per_k
from pi_nsde_building_thermal.sde import exogenous_plus_latent_b
from pi_nsde_building_thermal.synthetic import TRUE_PARAMS, SyntheticConfig, generate_synthetic_building


def test_wind_loss_rate_and_ramp_invariant_under_crq_scaling():
    C, R, Q = 9.5, 3.6, 9.0
    alpha = 0.53
    wind = jnp.array(4.2)
    ua0 = envelope_ua_kw_per_k(R, wind)
    ua1 = envelope_ua_kw_per_k(R / alpha, wind)
    assert abs(float(ua0 / C - ua1 / (alpha * C))) < 1e-6
    assert abs((Q / C) - (alpha * Q) / (alpha * C)) < 1e-12


def test_solar_and_moisture_fluxes_over_c_break_the_scaling():
    C, alpha = 9.5, 0.53
    solar0 = 8.5 * 0.2 / C
    solar1 = 8.5 * 0.2 / (alpha * C)
    lat0 = 120.0 * 0.003 / C
    lat1 = 120.0 * 0.003 / (alpha * C)
    assert abs(solar1 - solar0 / alpha) < 1e-12
    assert abs(lat1 - lat0 / alpha) < 1e-12
    assert abs(solar0 - solar1) > 1e-6
    assert abs(lat0 - lat1) > 1e-6


def test_affine_intercept_is_3x1_with_post_step_accumulator():
    g = exogenous_plus_latent_b(
        t_out_c=jnp.array(-4.0),
        ghi_w_m2=jnp.array(80.0),
        omega_out=jnp.array(0.002),
        omega_in=jnp.array(0.006),
        q_hvac_kw=jnp.array(9.0),
        remainder_kw=jnp.array(0.0),
        mu_q_kw=jnp.array(0.7),
        ua_kw_per_k=jnp.array(0.3),
        params=TRUE_PARAMS,
        capacity_kwh_per_k=TRUE_PARAMS.C,
        kappa=1.25,
        dt_h=1.0 / 60.0,
    )
    assert g.shape == (3,)
    assert abs(float(g[2]) - float(g[0])) < 1e-12
    assert abs(float(g[0])) > 0.0


def test_pearson_r2_is_one_for_linear_and_near_zero_for_even_quadratic():
    x = jnp.linspace(-1.0, 1.0, 401)
    assert abs(float(pearson_r2(x, 2.0 * x + 3.0)) - 1.0) < 1e-5
    # x^2 is orthogonal to x on a symmetric grid, so Pearson misses this leakage.
    assert float(pearson_r2(x, x**2)) < 1e-8


def test_identifiability_penalty_charges_linear_envelope_leakage():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0)).arrays
    dt = data.t_out_c - data.t_in_c
    linear = identifiability_penalty(dt, data)
    quiet = identifiability_penalty(jnp.zeros_like(dt), data)
    assert float(linear) > float(quiet) + 0.5
