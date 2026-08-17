"""Laplace UQ shapes and positive variances."""

import jax.numpy as jnp

from pinn_building.synthetic import SyntheticConfig, generate_synthetic_building
from pinn_building.train import TrainConfig
from pinn_building.uq import quantify_uncertainty


def test_uq_positive_variances_and_shapes():
    from pinn_building.model import init_params
    from pinn_building.train import filter_from_params

    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    import jax

    params = init_params(jax.random.PRNGKey(1), data.arrays, q_rated_mode="unknown")
    filt = filter_from_params(params, data.arrays, n_sub=5, q_rated_mode="unknown")
    cfg = TrainConfig(steps=8, n_sub=5, q_rated="unknown")
    uq = quantify_uncertainty(params, data.arrays, filt, cfg, n_sub=5, max_samples=4)
    assert "Q_rated" in uq.laplace.names
    assert uq.laplace.mean.shape == (9,)
    assert uq.laplace.sd.shape == (9,)
    assert jnp.all(uq.laplace.sd > 0)
    assert uq.t_sd_state.shape == data.arrays.t_in_c.shape
    assert jnp.all(uq.t_sd_state > 0)
    assert uq.t_q95.shape == data.arrays.t_in_c.shape
    assert float(jnp.mean(uq.t_q95 - uq.t_q05)) > 0


def test_known_mode_laplace_omits_q_rated():
    from pinn_building.model import init_params
    from pinn_building.train import filter_from_params
    import jax

    data = generate_synthetic_building(SyntheticConfig(days=1, seed=0))
    params = init_params(jax.random.PRNGKey(1), data.arrays, q_rated_mode="known")
    filt = filter_from_params(params, data.arrays, n_sub=5, q_rated_mode="known")
    cfg = TrainConfig(steps=8, n_sub=5, q_rated="known")
    uq = quantify_uncertainty(params, data.arrays, filt, cfg, n_sub=5, max_samples=2)
    assert "Q_rated" not in uq.laplace.names
    assert uq.laplace.mean.shape == (8,)
