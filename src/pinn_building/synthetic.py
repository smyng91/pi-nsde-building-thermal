"""Synthetic weather + SDE building plant + thermostat interval reports.

HVAC on/off is simulated with a hysteretic thermostat and exported as
interval runtime fraction (and, for eval / the optimistic known-kW protocol,
delivered mean power). The identifier never treats HVAC as a latent switching
process: on/off is observed; rated capacity may be known or learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pandas as pd
from jax import random

from pinn_building.physics import BuildingParams, humidity_ratio, saturation_vapor_pressure_pa
from pinn_building.sde import SdeNoise, simulate_em_step

TRUE_PARAMS = BuildingParams(C=9.5, R=3.6, A_s=8.5, beta=120.0, Q_rated=9.0)
TRUE_NOISE = SdeNoise(sigma_T=0.055, sigma_q=0.14, sigma_y=0.07, kappa=1.25)


class ChronologicalSplit(NamedTuple):
    """Train is a prefix; holdout is the contiguous suffix. Never shuffle timesteps."""

    n_train: int
    n_holdout: int
    n_total: int
    train_days: float
    holdout_days: float
    scheme: str


class Timeseries(NamedTuple):
    """Thermostat-interval arrays.

    ``hvac_on_frac`` is observed runtime in [0, 1]. ``q_hvac_kw`` is plant
    delivered power (eval-only in unknown-Q_rated mode; the optimistic
    known-kW protocol may use it). ``q_int_kw`` is hidden and must not be a
    feature, Kalman input, or loss target.
    """

    t_hours: jnp.ndarray
    t_out_c: jnp.ndarray
    ghi_w_m2: jnp.ndarray
    rh_out: jnp.ndarray
    wind_m_s: jnp.ndarray
    t_in_c: jnp.ndarray
    rh_in: jnp.ndarray
    omega_out: jnp.ndarray
    omega_in: jnp.ndarray
    q_hvac_kw: jnp.ndarray
    hvac_on_frac: jnp.ndarray
    q_int_kw: jnp.ndarray
    setpoint_c: jnp.ndarray


@dataclass(frozen=True)
class SyntheticConfig:
    days: int = 7
    report_dt_min: float = 5.0
    sim_dt_min: float = 1.0
    seed: int = 0
    latitude_deg: float = 41.8
    start_doy: int = 15
    t_in_init_c: float = 20.0
    heating_capacity_kw: float = 9.0
    deadband_k: float = 0.45
    indoor_rh: float = 0.40
    t_in_noise_k: float = 0.07
    weather_noise_k: float = 0.10
    q_int_init_kw: float = 0.6


@dataclass
class SyntheticDataset:
    frame: pd.DataFrame
    arrays: Timeseries
    true_params: BuildingParams
    true_noise: SdeNoise
    config: SyntheticConfig


def _ar1(key, n: int, tau_steps: float, std: float) -> jnp.ndarray:
    eps = random.normal(key, (n,))
    a = jnp.exp(-1.0 / tau_steps)
    sigma = std * jnp.sqrt(1.0 - a**2)

    def body(x, e):
        x = a * x + sigma * e
        return x, x

    _, xs = jax.lax.scan(body, 0.0, eps)
    return xs


def _ghi_clearsky(hour: jnp.ndarray, doy: jnp.ndarray, lat_deg: float) -> jnp.ndarray:
    lat = jnp.deg2rad(lat_deg)
    decl = jnp.deg2rad(23.45) * jnp.sin(2.0 * jnp.pi * (284.0 + doy) / 365.0)
    ha = jnp.deg2rad(15.0 * (hour - 12.0))
    sin_el = jnp.sin(lat) * jnp.sin(decl) + jnp.cos(lat) * jnp.cos(decl) * jnp.cos(ha)
    return 910.0 * jnp.maximum(sin_el, 0.0)


def _setpoint_c(t_hours: jnp.ndarray) -> jnp.ndarray:
    hour = t_hours % 24.0
    dow = jnp.floor(t_hours / 24.0) % 7.0
    weekend = dow >= 5.0
    occupied = jnp.where(
        weekend,
        (hour >= 8.0) & (hour < 23.0),
        (hour >= 7.0) & (hour < 22.0),
    )
    return jnp.where(occupied, 20.5, 17.5)


def occupancy_schedule_kw(t_hours: jnp.ndarray) -> jnp.ndarray:
    hour = t_hours % 24.0
    dow = jnp.floor(t_hours / 24.0) % 7.0
    weekend = dow >= 5.0
    morning = jnp.exp(-0.5 * ((hour - 7.5) / 1.1) ** 2)
    evening = jnp.exp(-0.5 * ((hour - 18.5) / 2.2) ** 2)
    midday = jnp.exp(-0.5 * ((hour - 12.5) / 1.6) ** 2)
    occ = 0.55 * morning + 1.0 * evening + jnp.where(weekend, 0.45 * midday, 0.12 * midday)
    return 0.35 + 1.15 * occ


def _weather(key, t_hours: jnp.ndarray, cfg: SyntheticConfig) -> dict[str, jnp.ndarray]:
    n = t_hours.shape[0]
    dt_h = float(t_hours[1] - t_hours[0])
    k1, k2, k3, k4 = random.split(key, 4)
    hour = t_hours % 24.0
    doy = cfg.start_doy + t_hours / 24.0
    t_syn = _ar1(k1, n, tau_steps=36.0 / dt_h, std=3.4)
    cloud = jnp.clip(0.32 + 0.28 * _ar1(k2, n, tau_steps=18.0 / dt_h, std=1.0), 0.0, 1.0)
    wind = jnp.maximum(
        3.2 + 1.6 * _ar1(k3, n, tau_steps=10.0 / dt_h, std=1.0) + 0.7 * jnp.sin(2 * jnp.pi * hour / 24.0),
        0.3,
    )
    t_diurnal = -6.2 * jnp.cos(2.0 * jnp.pi * (hour - 6.0) / 24.0)
    t_out = -2.0 + t_diurnal + t_syn
    ghi = _ghi_clearsky(hour, doy, cfg.latitude_deg) * (1.0 - 0.78 * cloud)
    dewpoint = -5.5 + 0.35 * t_syn + 0.8 * _ar1(k4, n, tau_steps=48.0 / dt_h, std=1.0)
    rh_out = jnp.clip(
        saturation_vapor_pressure_pa(dewpoint) / saturation_vapor_pressure_pa(t_out),
        0.25,
        0.95,
    )
    return {"t_out_c": t_out, "ghi_w_m2": ghi, "rh_out": rh_out, "wind_m_s": wind}


def _simulate_sde(key, weather: dict[str, jnp.ndarray], t_hours: jnp.ndarray, cfg: SyntheticConfig):
    dt_h = float(t_hours[1] - t_hours[0])
    n = t_hours.shape[0]
    k_t, k_q = random.split(key)
    dW_t = random.normal(k_t, (n,))
    dW_q = random.normal(k_q, (n,))
    setpoints = _setpoint_c(t_hours)
    mu_q = occupancy_schedule_kw(t_hours)
    t_out = weather["t_out_c"]
    ghi = weather["ghi_w_m2"]
    wind = weather["wind_m_s"]
    omega_out = humidity_ratio(t_out, weather["rh_out"])
    capacity = cfg.heating_capacity_kw
    deadband = cfg.deadband_k
    rh_in = cfg.indoor_rh

    def step(carry, inputs):
        t_in, q_int, heating_on = carry
        t_a, ghi_k, w_out, wind_k, t_set, mu, dw_t, dw_q = inputs
        heating_on = jnp.where(
            t_in >= t_set + deadband,
            False,
            jnp.where(t_in <= t_set - deadband, True, heating_on),
        )
        q_hvac = jnp.where(heating_on, capacity, 0.0)
        w_in = humidity_ratio(t_in, rh_in)
        t_next, q_next = simulate_em_step(
            t_in,
            q_int,
            t_a,
            ghi_k,
            w_out,
            w_in,
            q_hvac,
            wind_k,
            mu,
            TRUE_PARAMS,
            TRUE_NOISE,
            dt_h,
            dw_t,
            dw_q,
        )
        return (t_next, q_next, heating_on), (t_in, q_hvac, heating_on, w_in, q_int)

    inputs = (t_out, ghi, omega_out, wind, setpoints, mu_q, dW_t, dW_q)
    _, (t_in, q_hvac, on, omega_in, q_int) = jax.lax.scan(
        step, (cfg.t_in_init_c, cfg.q_int_init_kw, True), inputs
    )
    return t_in, q_hvac, on, omega_in, q_int


def _block_mean(x: jnp.ndarray, n_avg: int) -> jnp.ndarray:
    n = (x.shape[0] // n_avg) * n_avg
    return x[:n].reshape(-1, n_avg).mean(axis=1)


def generate_synthetic_building(config: SyntheticConfig | None = None) -> SyntheticDataset:
    """SDE plant + 5-minute thermostat averages, with HVAC runtime observed."""
    cfg = config or SyntheticConfig()
    sim_dt_h = cfg.sim_dt_min / 60.0
    n_sim = int(round(cfg.days * 24.0 / sim_dt_h))
    t_sim = jnp.arange(n_sim) * sim_dt_h
    key = random.PRNGKey(cfg.seed)
    key_w, key_s, key_n = random.split(key, 3)

    weather = _weather(key_w, t_sim, cfg)
    t_in, q_hvac, on, omega_in, q_int = _simulate_sde(key_s, weather, t_sim, cfg)
    omega_out = humidity_ratio(weather["t_out_c"], weather["rh_out"])

    n_avg = int(round(cfg.report_dt_min / cfg.sim_dt_min))
    t_hours = _block_mean(t_sim, n_avg)
    hvac_on_frac = _block_mean(on.astype(jnp.float32), n_avg)
    arrays = Timeseries(
        t_hours=t_hours,
        t_out_c=_block_mean(weather["t_out_c"], n_avg),
        ghi_w_m2=_block_mean(weather["ghi_w_m2"], n_avg),
        rh_out=_block_mean(weather["rh_out"], n_avg),
        wind_m_s=_block_mean(weather["wind_m_s"], n_avg),
        t_in_c=_block_mean(t_in, n_avg),
        rh_in=jnp.full_like(t_hours, cfg.indoor_rh),
        omega_out=_block_mean(omega_out, n_avg),
        omega_in=_block_mean(omega_in, n_avg),
        q_hvac_kw=_block_mean(q_hvac, n_avg),
        hvac_on_frac=hvac_on_frac,
        q_int_kw=_block_mean(q_int, n_avg),
        setpoint_c=_block_mean(_setpoint_c(t_sim), n_avg),
    )
    noise = random.normal(key_n, (t_hours.shape[0], 2))
    arrays = arrays._replace(
        t_in_c=arrays.t_in_c + cfg.t_in_noise_k * noise[:, 0],
        t_out_c=arrays.t_out_c + cfg.weather_noise_k * noise[:, 1],
    )

    interval_s = cfg.report_dt_min * 60.0
    t_np = __import__("numpy").asarray(t_hours)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-15") + pd.to_timedelta(t_np, unit="h"),
            "t_hours": t_hours,
            "t_out_c": arrays.t_out_c,
            "ghi_w_m2": arrays.ghi_w_m2,
            "rh_out_frac": arrays.rh_out,
            "wind_m_s": arrays.wind_m_s,
            "t_in_c": arrays.t_in_c,
            "rh_in_frac": arrays.rh_in,
            "heat_setpoint_c": arrays.setpoint_c,
            "hvac_kw": arrays.q_hvac_kw,
            "hvac_on_frac": arrays.hvac_on_frac,
            "hvac_runtime_s": arrays.hvac_on_frac * interval_s,
            # Evaluation-only; the identifier must not read this column.
            "q_int_kw_hidden": arrays.q_int_kw,
        }
    )
    return SyntheticDataset(
        frame=frame,
        arrays=arrays,
        true_params=TRUE_PARAMS._replace(Q_rated=cfg.heating_capacity_kw),
        true_noise=TRUE_NOISE,
        config=cfg,
    )


def slice_timeseries(data: Timeseries, start: int, stop: int | None = None) -> Timeseries:
    sl = slice(start, stop)
    return Timeseries(*(getattr(data, name)[sl] for name in Timeseries._fields))


def chronological_split(
    data: Timeseries,
    holdout_days: float | None = None,
    holdout_frac: float | None = None,
) -> tuple[Timeseries, Timeseries, ChronologicalSplit]:
    """Last block holdout. Default: last 2 days if the series is long enough, else last 30%.

    Never shuffles, never draws random rows, never puts future samples in train.
    """
    n = int(data.t_hours.shape[0])
    if n < 4:
        raise ValueError("Need at least 4 timesteps to split train/holdout.")
    dt_h = float(data.t_hours[1] - data.t_hours[0])
    span_h = float(data.t_hours[-1] - data.t_hours[0]) + dt_h

    if holdout_days is None and holdout_frac is None:
        if span_h >= 5.0 * 24.0:
            holdout_days = 2.0
        else:
            holdout_frac = 0.30

    if holdout_days is not None:
        n_hold = int(round(float(holdout_days) * 24.0 / dt_h))
        scheme = f"chronological_last_{float(holdout_days):g}_days"
    else:
        n_hold = int(round(n * float(holdout_frac)))
        scheme = f"chronological_last_{float(holdout_frac):.0%}"

    n_hold = max(1, min(n_hold, n - 2))
    n_train = n - n_hold
    train = slice_timeseries(data, 0, n_train)
    holdout = slice_timeseries(data, n_train, n)
    info = ChronologicalSplit(
        n_train=n_train,
        n_holdout=n_hold,
        n_total=n,
        train_days=n_train * dt_h / 24.0,
        holdout_days=n_hold * dt_h / 24.0,
        scheme=scheme,
    )
    return train, holdout, info
