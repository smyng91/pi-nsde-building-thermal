"""Two-stage MAP of the interval-average Kalman likelihood (train only).

Stage A: neural remainder frozen at 0; fit {C, R, Q_rated (if unknown), A_s, β, σ, κ, Fourier μ_q}.
Stage B: unfreeze remainder (identifiability penalty), smaller LR; optionally
freeze C, R, and Q_rated first, then jointly fine-tune.

Indoor T Kalman tracking is not a success metric. Holdout scoring is open-loop.
HVAC runtime is observed and signed (heating positive, cooling negative); rated
capacity is learned only in unknown-Q_rated mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from pi_nsde_building_thermal.model import (
    ModelParams,
    canonicalize_hvac,
    decode_building,
    decode_noise,
    identifiability_penalty,
    init_params,
    normalize_hvac_mode,
    normalize_q_rated_mode,
    occupancy_regularizer,
    observed_hvac_kw,
    remainder_and_sigma_scale,
    weak_prior,
)
from pi_nsde_building_thermal.physics import BuildingParams
from pi_nsde_building_thermal.sde import (
    FilterResult,
    OpenLoopResult,
    interval_average_kalman,
    interval_average_open_loop,
    occupancy_mean_kw,
)
from pi_nsde_building_thermal.synthetic import ChronologicalSplit, Timeseries, chronological_split

_HISTORY_KEYS = (
    "step",
    "stage",
    "loss",
    "nll",
    "ident",
    "occ",
    "remainder_rms",
    "C",
    "R",
    "Q_rated",
    "A_s",
    "beta",
)


@dataclass
class TrainConfig:
    """If ``steps`` is set (tests/quick), only Stage A runs for that many updates."""

    steps: int | None = None
    stage_a_steps: int = 1800
    stage_b_freeze_cr_steps: int = 300
    stage_b_joint_steps: int = 1400
    learning_rate: float = 3.2e-3
    stage_b_lr: float = 1.15e-3
    seed: int = 1
    log_every: int = 100
    n_sub: int = 5
    lambda_id: float = 0.15
    lambda_id_b: float = 0.45
    lambda_prior: float = 0.002
    lambda_occ: float = 0.05
    q_rated: str = "unknown"
    hvac_mode: str = "auto"
    prior: BuildingParams = field(
        default_factory=lambda: BuildingParams(C=6.0, R=6.0, A_s=4.0, beta=80.0, Q_rated=6.0)
    )

    def __post_init__(self):
        self.q_rated = normalize_q_rated_mode(self.q_rated)
        self.hvac_mode = normalize_hvac_mode(self.hvac_mode)

    def resolved_stage_steps(self) -> tuple[int, int, int]:
        if self.steps is not None:
            return int(self.steps), 0, 0
        return self.stage_a_steps, self.stage_b_freeze_cr_steps, self.stage_b_joint_steps


@dataclass
class TrainResult:
    params: ModelParams
    history: dict[str, list]
    filter: FilterResult
    estimated: BuildingParams
    n_sub: int
    dt_sub_h: float
    remainder_gate: float
    lambda_id: float
    n_train: int
    q_rated: str = "unknown"
    hvac_mode: str = "auto"


@dataclass
class IdentificationResult:
    split: ChronologicalSplit
    train: Timeseries
    holdout: Timeseries
    fit: TrainResult
    holdout_open_loop: OpenLoopResult
    holdout_rmse: float
    holdout_mae: float
    train_nll_mean: float
    train_nll_sum: float


def filter_from_params(
    params: ModelParams,
    data: Timeseries,
    n_sub: int,
    remainder_gate: float = 1.0,
    q_rated_mode: str = "unknown",
) -> FilterResult:
    dt_report = data.t_hours[1] - data.t_hours[0]
    dt_sub = dt_report / n_sub
    building = decode_building(params.phys)
    noise = decode_noise(params.phys)
    remainder, sig_scale = remainder_and_sigma_scale(
        params,
        data,
        remainder_gate=remainder_gate,
        sigma_gate=remainder_gate,
        q_rated_mode=q_rated_mode,
    )
    mu_q = occupancy_mean_kw(data.t_hours, params.phys.fourier_q)
    q_hvac = observed_hvac_kw(params, data, q_rated_mode)
    return interval_average_kalman(
        building,
        noise,
        remainder,
        sig_scale,
        mu_q,
        data.t_out_c,
        data.ghi_w_m2,
        data.omega_out,
        data.omega_in,
        q_hvac,
        data.wind_m_s,
        data.t_in_c,
        n_sub,
        dt_sub,
        t0=data.t_in_c[0],
        q0=0.7,
    )


def open_loop_from_params(
    params: ModelParams,
    data: Timeseries,
    n_sub: int,
    t0: float,
    q0: float,
    remainder_gate: float = 1.0,
    q_rated_mode: str = "unknown",
) -> OpenLoopResult:
    """Mean rollout using weather + signed HVAC runtime (× estimated Q_rated if unknown).

    Does not read T_in, Q_int, or (in unknown mode) delivered q_hvac_kw.
    """
    dt_report = data.t_hours[1] - data.t_hours[0]
    dt_sub = dt_report / n_sub
    building = decode_building(params.phys)
    noise = decode_noise(params.phys)
    remainder, _ = remainder_and_sigma_scale(
        params,
        data,
        remainder_gate=remainder_gate,
        sigma_gate=remainder_gate,
        q_rated_mode=q_rated_mode,
    )
    mu_q = occupancy_mean_kw(data.t_hours, params.phys.fourier_q)
    q_hvac = observed_hvac_kw(params, data, q_rated_mode)
    return interval_average_open_loop(
        building,
        remainder,
        mu_q,
        data.t_out_c,
        data.ghi_w_m2,
        data.omega_out,
        data.omega_in,
        q_hvac,
        data.wind_m_s,
        n_sub,
        dt_sub,
        t0,
        q0,
        kappa=noise.kappa,
    )


def total_loss(
    params: ModelParams,
    data: Timeseries,
    cfg: TrainConfig,
    remainder_gate: float = 1.0,
    lambda_id: float | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Mean-interval NLL + penalties (optimizer scale). Not the Laplace objective."""
    lam = cfg.lambda_id if lambda_id is None else lambda_id
    mode = cfg.q_rated
    filt = filter_from_params(params, data, cfg.n_sub, remainder_gate=remainder_gate, q_rated_mode=mode)
    remainder, _ = remainder_and_sigma_scale(
        params, data, remainder_gate=remainder_gate, sigma_gate=remainder_gate, q_rated_mode=mode
    )
    nll = filt.nll / data.t_in_c.shape[0]
    ident = identifiability_penalty(remainder, data, q_rated_mode=mode)
    prior = weak_prior(params.phys, cfg.prior)
    occ = occupancy_regularizer(params.phys)
    loss = nll + lam * ident + cfg.lambda_prior * prior + cfg.lambda_occ * occ
    b = decode_building(params.phys)
    aux = {
        "nll": nll,
        "nll_sum": filt.nll,
        "ident": ident,
        "occ": occ,
        "remainder_rms": jnp.sqrt(jnp.mean(remainder**2)),
        "C": b.C,
        "R": b.R,
        "Q_rated": b.Q_rated,
        "A_s": b.A_s,
        "beta": b.beta,
    }
    return loss, aux


def map_objective_sum(
    params: ModelParams,
    data: Timeseries,
    cfg: TrainConfig,
    remainder_gate: float = 1.0,
    lambda_id: float | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Train MAP objective with **sum** of interval NLLs — use this for Laplace UQ.

    Same critical points as ``total_loss`` (mean NLL) because
    J_sum = N * J_mean. Hessian(J_sum) is the observed information scale.
    """
    mean_loss, aux = total_loss(params, data, cfg, remainder_gate=remainder_gate, lambda_id=lambda_id)
    n = data.t_in_c.shape[0]
    return n * mean_loss, aux


def _trainable_mask(
    params: ModelParams,
    *,
    freeze_remainder: bool,
    freeze_sigma_net: bool,
    freeze_cr: bool,
    freeze_q_rated: bool,
) -> ModelParams:
    ones = jax.tree.map(lambda x: jnp.ones_like(x), params)
    phys = ones.phys
    if freeze_cr:
        phys = phys._replace(
            raw_C=jnp.zeros_like(params.phys.raw_C),
            raw_R=jnp.zeros_like(params.phys.raw_R),
        )
    if freeze_q_rated:
        phys = phys._replace(raw_Q_rated=jnp.zeros_like(params.phys.raw_Q_rated))
    rem = jax.tree.map(lambda x: jnp.zeros_like(x), params.remainder_net) if freeze_remainder else ones.remainder_net
    sig = jax.tree.map(lambda x: jnp.zeros_like(x), params.sigma_net) if freeze_sigma_net else ones.sigma_net
    return ModelParams(
        phys=phys,
        remainder_net=rem,
        sigma_net=sig,
        feat_mean=jnp.zeros_like(params.feat_mean),
        feat_std=jnp.zeros_like(params.feat_std),
    )


def _run_stage(
    params: ModelParams,
    data: Timeseries,
    cfg: TrainConfig,
    *,
    steps: int,
    learning_rate: float,
    remainder_gate: float,
    lambda_id: float,
    freeze_remainder: bool,
    freeze_sigma_net: bool,
    freeze_cr: bool,
    freeze_q_rated: bool,
    stage_name: str,
    verbose: bool,
    cosine: bool,
) -> tuple[ModelParams, dict[str, list]]:
    empty = {k: [] for k in _HISTORY_KEYS}
    if steps <= 0:
        return params, empty

    mask = _trainable_mask(
        params,
        freeze_remainder=freeze_remainder,
        freeze_sigma_net=freeze_sigma_net,
        freeze_cr=freeze_cr,
        freeze_q_rated=freeze_q_rated,
    )
    if cosine:
        lr = optax.cosine_decay_schedule(learning_rate, max(steps, 1), alpha=0.55)
    else:
        lr = learning_rate
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    opt_state = optimizer.init(params)
    gate = float(remainder_gate)
    lam = float(lambda_id)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            return total_loss(p, data, cfg, remainder_gate=gate, lambda_id=lam)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree.map(lambda g, m: g * m, grads, mask)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    history = {k: [] for k in empty}
    for i in range(steps):
        params, opt_state, loss, aux = step(params, opt_state)
        if i % cfg.log_every == 0 or i == steps - 1:
            row = {
                "step": i,
                "stage": stage_name,
                "loss": float(loss),
                "nll": float(aux["nll"]),
                "ident": float(aux["ident"]),
                "occ": float(aux["occ"]),
                "remainder_rms": float(aux["remainder_rms"]),
                "C": float(aux["C"]),
                "R": float(aux["R"]),
                "Q_rated": float(aux["Q_rated"]),
                "A_s": float(aux["A_s"]),
                "beta": float(aux["beta"]),
            }
            for k, v in row.items():
                history[k].append(v)
            if verbose:
                q_txt = f"  Qr={row['Q_rated']:.3f}" if cfg.q_rated == "unknown" else ""
                print(
                    f"{stage_name} {i:5d}/{steps}  loss={row['loss']:.4f}  nll={row['nll']:.4f}  "
                    f"C={row['C']:.3f}  R={row['R']:.3f}{q_txt}  r_rms={row['remainder_rms']:.4f}"
                )
    return params, history


def _merge_histories(parts: list[dict[str, list]]) -> dict[str, list]:
    out = {k: [] for k in _HISTORY_KEYS}
    for hist in parts:
        for k in _HISTORY_KEYS:
            out[k].extend(hist[k])
    return out


def train_sde(data: Timeseries, config: TrainConfig | None = None, verbose: bool = True) -> TrainResult:
    """Fit on the provided series only. Pass the **train** slice; do not pass holdout."""
    cfg = config or TrainConfig()
    data = canonicalize_hvac(data, cfg.hvac_mode)
    n_a, n_b1, n_b2 = cfg.resolved_stage_steps()
    params = init_params(jax.random.PRNGKey(cfg.seed), data, prior=cfg.prior, q_rated_mode=cfg.q_rated)

    if verbose:
        hvac_obs = (
            "HVAC = Q_rated * signed runtime (capacity unknown)"
            if cfg.q_rated == "unknown"
            else "HVAC = metered q_hvac_kw (capacity known)"
        )
        print(
            f"Two-stage ID on n={int(data.t_in_c.shape[0])} train steps | "
            f"A={n_a} (remainder=0), B_freeze_CR={n_b1}, B_joint={n_b2} | "
            f"{hvac_obs} | hvac_mode={cfg.hvac_mode}"
        )

    freeze_q_known = cfg.q_rated == "known"
    params, h_a = _run_stage(
        params,
        data,
        cfg,
        steps=n_a,
        learning_rate=cfg.learning_rate,
        remainder_gate=0.0,
        lambda_id=cfg.lambda_id,
        freeze_remainder=True,
        freeze_sigma_net=True,
        freeze_cr=False,
        freeze_q_rated=freeze_q_known,
        stage_name="A",
        verbose=verbose,
        cosine=False,
    )
    params, h_b1 = _run_stage(
        params,
        data,
        cfg,
        steps=n_b1,
        learning_rate=cfg.stage_b_lr,
        remainder_gate=1.0,
        lambda_id=cfg.lambda_id_b,
        freeze_remainder=False,
        freeze_sigma_net=False,
        freeze_cr=True,
        freeze_q_rated=True,
        stage_name="B1",
        verbose=verbose,
        cosine=True,
    )
    params, h_b2 = _run_stage(
        params,
        data,
        cfg,
        steps=n_b2,
        learning_rate=cfg.stage_b_lr,
        remainder_gate=1.0,
        lambda_id=cfg.lambda_id_b,
        freeze_remainder=False,
        freeze_sigma_net=False,
        freeze_cr=False,
        freeze_q_rated=freeze_q_known,
        stage_name="B2",
        verbose=verbose,
        cosine=True,
    )

    remainder_gate = 1.0 if (n_b1 + n_b2) > 0 else 0.0
    lambda_id = cfg.lambda_id_b if remainder_gate > 0.5 else cfg.lambda_id
    filt = filter_from_params(
        params, data, cfg.n_sub, remainder_gate=remainder_gate, q_rated_mode=cfg.q_rated
    )
    dt_report = float(data.t_hours[1] - data.t_hours[0])
    return TrainResult(
        params=params,
        history=_merge_histories([h_a, h_b1, h_b2]),
        filter=filt,
        estimated=decode_building(params.phys),
        n_sub=cfg.n_sub,
        dt_sub_h=dt_report / cfg.n_sub,
        remainder_gate=remainder_gate,
        lambda_id=lambda_id,
        n_train=int(data.t_in_c.shape[0]),
        q_rated=cfg.q_rated,
        hvac_mode=cfg.hvac_mode,
    )


def temperature_rmse_mae(y_pred, y_obs) -> tuple[float, float]:
    err = jnp.asarray(y_pred) - jnp.asarray(y_obs)
    rmse = float(jnp.sqrt(jnp.mean(err**2)))
    mae = float(jnp.mean(jnp.abs(err)))
    return rmse, mae


def identify_building(
    arrays: Timeseries,
    config: TrainConfig | None = None,
    holdout_days: float | None = None,
    holdout_frac: float | None = None,
    verbose: bool = True,
) -> IdentificationResult:
    """Chronological split + two-stage train-only fit + holdout open-loop T metric."""
    cfg = config or TrainConfig()
    arrays = canonicalize_hvac(arrays, cfg.hvac_mode)
    train, holdout, split = chronological_split(
        arrays, holdout_days=holdout_days, holdout_frac=holdout_frac
    )
    if verbose:
        print(
            f"Chronological split ({split.scheme}): "
            f"train [0, {split.n_train}) = {split.train_days:.2f} d, "
            f"holdout [{split.n_train}, {split.n_total}) = {split.holdout_days:.2f} d. "
            "Holdout is not used to fit C, R, Q_rated, remainder, Fourier mu_q, or noise."
        )
    fit = train_sde(train, cfg, verbose=verbose)
    ol = open_loop_from_params(
        fit.params,
        holdout,
        fit.n_sub,
        t0=float(fit.filter.t_mean[-1]),
        q0=float(fit.filter.q_mean[-1]),
        remainder_gate=fit.remainder_gate,
        q_rated_mode=fit.q_rated,
    )
    rmse, mae = temperature_rmse_mae(ol.y_pred, holdout.t_in_c)
    nll_sum = float(fit.filter.nll)
    nll_mean = nll_sum / max(split.n_train, 1)
    if verbose:
        if fit.q_rated == "unknown":
            hvac_txt = "weather + estimated Q_rated × signed holdout runtime"
        else:
            hvac_txt = "weather + known HVAC kW"
        print(
            f"Holdout open-loop T ({hvac_txt}, frozen MAP): "
            f"RMSE={rmse:.3f} K  MAE={mae:.3f} K"
        )
        print(f"Secondary train mean Kalman NLL={nll_mean:.4f} (not a T-accuracy claim).")
    return IdentificationResult(
        split=split,
        train=train,
        holdout=holdout,
        fit=fit,
        holdout_open_loop=ol,
        holdout_rmse=rmse,
        holdout_mae=mae,
        train_nll_mean=nll_mean,
        train_nll_sum=nll_sum,
    )
