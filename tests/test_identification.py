"""No-leakage identification protocol: split, hidden Q_int, two-stage remainder, UQ."""

import jax
import jax.numpy as jnp

from pi_nsde_building_thermal.constants import FILTER_Q0_KW
from pi_nsde_building_thermal.model import (
    decode_building,
    exogenous_features,
    init_params,
    remainder_and_sigma_scale,
)
from pi_nsde_building_thermal.physics import BuildingParams
from pi_nsde_building_thermal.synthetic import SyntheticConfig, chronological_split, generate_synthetic_building
from pi_nsde_building_thermal.train import (
    TrainConfig,
    filter_from_params,
    map_objective_sum,
    open_loop_from_params,
    total_loss,
    train_sde,
)
from pi_nsde_building_thermal.uq import quantify_uncertainty


def test_hidden_q_int_not_in_features_or_filter():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    params = init_params(jax.random.PRNGKey(1), data.arrays)
    feat = exogenous_features(data.arrays, q_rated_mode="unknown")
    q = data.arrays.q_int_kw
    t_in = data.arrays.t_in_c
    for i in range(feat.shape[1]):
        assert not jnp.allclose(feat[:, i], q, atol=1e-5)
        assert not jnp.allclose(feat[:, i], t_in, atol=1e-5)
    assert jnp.allclose(feat[:, 4], data.arrays.hvac_on_frac)
    assert not jnp.allclose(feat[:, 4], data.arrays.q_hvac_kw, atol=1e-3)

    filt_a = filter_from_params(params, data.arrays, 5)
    scrambled = data.arrays._replace(q_int_kw=q + 3.5)
    filt_b = filter_from_params(params, scrambled, 5)
    assert jnp.allclose(filt_a.nll, filt_b.nll)
    assert jnp.allclose(filt_a.t_mean, filt_b.t_mean)

    r1, s1 = remainder_and_sigma_scale(params, data.arrays)
    r2, s2 = remainder_and_sigma_scale(params, scrambled)
    assert jnp.allclose(r1, r2)
    assert jnp.allclose(s1, s2)


def test_default_seven_day_split_is_last_two_days():
    data = generate_synthetic_building(SyntheticConfig(days=7, seed=0))
    _, _, split = chronological_split(data.arrays)
    assert split.n_train + split.n_holdout == data.arrays.t_hours.shape[0]
    assert abs(split.holdout_days - 2.0) < 0.02
    assert float(data.arrays.t_hours[split.n_train - 1]) < float(data.arrays.t_hours[split.n_train])


def test_holdout_indices_are_after_train():
    data = generate_synthetic_building(SyntheticConfig(days=3, seed=0))
    train, holdout, split = chronological_split(data.arrays, holdout_days=1.0)
    assert split.n_train + split.n_holdout == data.arrays.t_hours.shape[0]
    assert float(train.t_hours[-1]) < float(holdout.t_hours[0])
    assert jnp.allclose(data.arrays.t_hours[: split.n_train], train.t_hours)
    assert jnp.allclose(data.arrays.t_hours[split.n_train :], holdout.t_hours)
    assert split.scheme.startswith("chronological")


def test_stage_a_remainder_starts_at_zero():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    train, _, _ = chronological_split(data.arrays, holdout_frac=0.3)
    result = train_sde(train, TrainConfig(steps=6, log_every=6, seed=1, q_rated="unknown"), verbose=False)
    assert result.remainder_gate == 0.0
    assert result.q_rated == "unknown"
    r_gated, _ = remainder_and_sigma_scale(
        result.params, train, remainder_gate=0.0, q_rated_mode="unknown"
    )
    assert float(jnp.max(jnp.abs(r_gated))) < 1e-8
    r_open, _ = remainder_and_sigma_scale(
        result.params, train, remainder_gate=1.0, q_rated_mode="unknown"
    )
    assert float(jnp.max(jnp.abs(r_open))) < 1e-6


def test_uq_uses_train_only_and_sum_nll():
    data = generate_synthetic_building(SyntheticConfig(days=2, seed=0))
    train, holdout, split = chronological_split(data.arrays, holdout_frac=0.3)
    params = init_params(jax.random.PRNGKey(1), train)
    cfg = TrainConfig(steps=4, n_sub=5)
    filt = filter_from_params(params, train, n_sub=5, remainder_gate=0.0)
    uq = quantify_uncertainty(
        params, train, filt, cfg, n_sub=5, max_samples=2, remainder_gate=0.0
    )
    assert uq.t_mean.shape[0] == split.n_train
    assert uq.laplace.n_obs == split.n_train
    assert "sum_NLL" in uq.laplace.method or "sum_nll" in uq.laplace.method.lower()

    mean_loss, _ = total_loss(params, train, cfg, remainder_gate=0.0)
    sum_loss, _ = map_objective_sum(params, train, cfg, remainder_gate=0.0)
    n = int(train.t_in_c.shape[0])
    assert abs(float(sum_loss) - float(mean_loss) * n) < 1e-3 * max(n, 1)

    ol = open_loop_from_params(
        params, holdout, 5, t0=20.0, q0=FILTER_Q0_KW, remainder_gate=0.0, q_rated_mode="unknown"
    )
    scrambled = holdout._replace(t_in_c=holdout.t_in_c + 8.0, q_int_kw=holdout.q_int_kw + 4.0)
    ol2 = open_loop_from_params(
        params, scrambled, 5, t0=20.0, q0=FILTER_Q0_KW, remainder_gate=0.0, q_rated_mode="unknown"
    )
    assert jnp.allclose(ol.y_pred, ol2.y_pred)


def test_init_starts_at_config_prior_not_plant():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    prior = BuildingParams(C=4.0, R=9.0, A_s=4.0, beta=80.0, Q_rated=4.0)
    params = init_params(jax.random.PRNGKey(1), data.arrays, prior=prior)
    b = decode_building(params.phys)
    assert abs(float(b.C) - 4.0) < 1e-4
    assert abs(float(b.R) - 9.0) < 1e-4
    assert abs(float(b.Q_rated) - 4.0) < 1e-4


def test_exported_omega_matches_observed_interval_temperature():
    from pi_nsde_building_thermal.physics import humidity_ratio

    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    w_in = humidity_ratio(data.arrays.t_in_c, data.arrays.rh_in)
    w_out = humidity_ratio(data.arrays.t_out_c, data.arrays.rh_out)
    assert jnp.allclose(data.arrays.omega_in, w_in, atol=1e-5)
    assert jnp.allclose(data.arrays.omega_out, w_out, atol=1e-5)


def test_beta_stays_at_prior_when_not_learned():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    train, _, _ = chronological_split(data.arrays, holdout_frac=0.3)
    result = train_sde(
        train,
        TrainConfig(steps=12, log_every=12, seed=1, q_rated="unknown", learn_beta=False),
        verbose=False,
    )
    assert abs(float(result.estimated.beta) - 120.0) < 1e-3



def test_graybox_keeps_remainder_off_after_stage_b():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    train, _, _ = chronological_split(data.arrays, holdout_frac=0.3)
    result = train_sde(
        train,
        TrainConfig(
            stage_a_steps=4,
            stage_b_freeze_cr_steps=2,
            stage_b_joint_steps=2,
            log_every=8,
            seed=1,
            q_rated="unknown",
            neural_remainder=False,
        ),
        verbose=False,
    )
    assert result.remainder_gate == 0.0
    r, _ = remainder_and_sigma_scale(
        result.params, train, remainder_gate=result.remainder_gate, q_rated_mode="unknown"
    )
    assert float(jnp.max(jnp.abs(r))) < 1e-8
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    params = init_params(jax.random.PRNGKey(1), data.arrays, q_rated_mode="unknown")
    q0 = float(decode_building(params.phys).Q_rated)
    assert abs(q0 - 9.0) > 1.5
    assert abs(q0 - 6.0) < 0.05


def test_unknown_mode_ignores_q_hvac_kw_but_uses_on_frac():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0)).arrays
    params = init_params(jax.random.PRNGKey(1), data, q_rated_mode="unknown")
    cfg = TrainConfig(steps=2, n_sub=5, q_rated="unknown")

    nll0 = filter_from_params(params, data, 5, remainder_gate=0.0, q_rated_mode="unknown").nll
    loss0, _ = total_loss(params, data, cfg, remainder_gate=0.0)

    kw_scrambled = data._replace(q_hvac_kw=data.q_hvac_kw * 0.0 + 3.7)
    nll_kw = filter_from_params(params, kw_scrambled, 5, remainder_gate=0.0, q_rated_mode="unknown").nll
    loss_kw, _ = total_loss(params, kw_scrambled, cfg, remainder_gate=0.0)
    feat0 = exogenous_features(data, q_rated_mode="unknown")
    feat_kw = exogenous_features(kw_scrambled, q_rated_mode="unknown")
    r0, s0 = remainder_and_sigma_scale(params, data, q_rated_mode="unknown")
    r_kw, s_kw = remainder_and_sigma_scale(params, kw_scrambled, q_rated_mode="unknown")
    assert jnp.allclose(nll0, nll_kw)
    assert jnp.allclose(loss0, loss_kw)
    assert jnp.allclose(feat0, feat_kw)
    assert jnp.allclose(r0, r_kw)
    assert jnp.allclose(s0, s_kw)

    on_scrambled = data._replace(hvac_on_frac=jnp.clip(1.0 - data.hvac_on_frac, 0.0, 1.0))
    nll_on = filter_from_params(params, on_scrambled, 5, remainder_gate=0.0, q_rated_mode="unknown").nll
    assert not jnp.allclose(nll0, nll_on, rtol=0.0, atol=1e-5)

    ol0 = open_loop_from_params(params, data, 5, t0=20.0, q0=FILTER_Q0_KW, remainder_gate=0.0, q_rated_mode="unknown")
    ol_kw = open_loop_from_params(
        params, kw_scrambled, 5, t0=20.0, q0=FILTER_Q0_KW, remainder_gate=0.0, q_rated_mode="unknown"
    )
    ol_on = open_loop_from_params(
        params, on_scrambled, 5, t0=20.0, q0=FILTER_Q0_KW, remainder_gate=0.0, q_rated_mode="unknown"
    )
    assert jnp.allclose(ol0.y_pred, ol_kw.y_pred)
    assert not jnp.allclose(ol0.y_pred, ol_on.y_pred, rtol=0.0, atol=1e-5)


def test_known_mode_reads_q_hvac_kw():
    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0)).arrays
    params = init_params(jax.random.PRNGKey(1), data, q_rated_mode="known")
    nll0 = filter_from_params(params, data, 5, remainder_gate=0.0, q_rated_mode="known").nll
    kw_scrambled = data._replace(q_hvac_kw=data.q_hvac_kw * 0.0 + 3.7)
    nll_kw = filter_from_params(params, kw_scrambled, 5, remainder_gate=0.0, q_rated_mode="known").nll
    assert not jnp.allclose(nll0, nll_kw, rtol=0.0, atol=1e-5)


def test_unknown_holdout_stays_chronological_last_two_of_seven():
    data = generate_synthetic_building(SyntheticConfig(days=7, seed=0))
    train, holdout, split = chronological_split(data.arrays)
    assert split.scheme.startswith("chronological")
    assert abs(split.holdout_days - 2.0) < 0.02
    assert float(train.t_hours[-1]) < float(holdout.t_hours[0])
    cfg = TrainConfig(q_rated="unknown")
    assert cfg.q_rated == "unknown"
