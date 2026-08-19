"""CSV timeseries I/O and fitted-parameter checkpoints.

Custom files need indoor temperature, outdoor temperature, and HVAC runtime
(heating, cooling, or signed). Delivered HVAC kilowatts are optional and unused
in the default unknown-Q_rated protocol. Hidden occupancy must not be a required
column.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pi_nsde_building_thermal.model import ModelParams, canonicalize_hvac, decode_building, decode_noise
from pi_nsde_building_thermal.physics import humidity_ratio
from pi_nsde_building_thermal.synthetic import Timeseries

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "datetime", "date"),
    "t_hours": ("t_hours", "hours", "hour"),
    "t_in_c": (
        "t_in_c",
        "t_in",
        "indoor_temp",
        "indoor_temperature",
        "thermostat_temperature",
        "current_temp",
    ),
    "t_out_c": (
        "t_out_c",
        "t_out",
        "outdoor_temp",
        "outdoor_temperature",
        "ambient_temp",
        "dry_bulb",
    ),
    "ghi_w_m2": ("ghi_w_m2", "ghi", "irradiance", "solar", "global_horizontal"),
    "rh_out": ("rh_out_frac", "rh_out", "outdoor_humidity", "rh_outdoor"),
    "rh_in": ("rh_in_frac", "rh_in", "indoor_humidity", "rh_indoor"),
    "wind_m_s": ("wind_m_s", "wind", "wind_speed"),
    "heating_on_frac": (
        "heating_on_frac",
        "heating_on",
        "aux_heat_frac",
        "heat_on_frac",
        "heat_on",
    ),
    "cooling_on_frac": (
        "cooling_on_frac",
        "cooling_on",
        "cool_on_frac",
        "cool_on",
        "comp_cool_frac",
        "compressor_cool",
    ),
    "hvac_on_frac": (
        "hvac_on_frac",
        "runtime_frac",
        "hvac_on",
        "equipment_running",
    ),
    "hvac_runtime_s": ("hvac_runtime_s", "runtime_s"),
    "heat_runtime_s": ("heat_runtime_s", "heating_runtime_s"),
    "cool_runtime_s": ("cool_runtime_s", "cooling_runtime_s"),
    "q_hvac_kw": ("q_hvac_kw", "hvac_kw", "hvac_power_kw"),
    "setpoint_c": (
        "heat_setpoint_c",
        "cool_setpoint_c",
        "setpoint_c",
        "setpoint",
        "heat_set_temp",
        "cool_set_temp",
    ),
    "q_int_kw": ("q_int_kw_hidden", "q_int_kw"),
}


def _find_column(columns: pd.Index, keys: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def _series(frame: pd.DataFrame, logical: str, default: float | None = None) -> np.ndarray:
    col = _find_column(frame.columns, _COLUMN_ALIASES[logical])
    if col is None:
        if default is None:
            raise ValueError(
                f"CSV is missing {logical}. Accepted names: {_COLUMN_ALIASES[logical]}"
            )
        return np.full(len(frame), default, dtype=np.float32)
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).all() and default is not None:
        return np.full(len(frame), default, dtype=np.float32)
    if np.isnan(values).any():
        fill = default if default is not None else float(np.nanmedian(values))
        values = np.where(np.isnan(values), fill, values)
    return values.astype(np.float32)


def _as_fraction(values: np.ndarray) -> np.ndarray:
    """Clip a humidity or similar fraction to [0, 1]. Percents are divided by 100."""
    out = values.astype(np.float32)
    if np.nanmax(out) > 1.5:
        out = out / 100.0
    return np.clip(out, 0.0, 1.0)


def _as_runtime_frac(values: np.ndarray, *, signed: bool = False) -> np.ndarray:
    """Runtime fraction. Signed channels live in [-1, 1]; unsigned in [0, 1]."""
    out = np.asarray(values, dtype=np.float64)
    out = np.where(np.isnan(out), 0.0, out)
    peak = np.nanmax(np.abs(out)) if out.size else 0.0
    if peak > 1.5:
        out = out / 100.0
    if signed or (out.size and float(np.min(out)) < -1e-8):
        return np.clip(out, -1.0, 1.0).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _series_to_on_frac(raw_on: pd.Series, *, signed: bool = False) -> np.ndarray:
    if raw_on.dtype == bool:
        return raw_on.astype(np.float32).to_numpy()
    if not pd.api.types.is_numeric_dtype(raw_on):
        lowered = raw_on.dropna().astype(str).str.lower()
        if lowered.isin({"true", "false", "yes", "no"}).all():
            mapped = raw_on.map(lambda x: 1.0 if str(x).lower() in {"true", "yes", "1"} else 0.0)
            return mapped.to_numpy(dtype=np.float32)
    return _as_runtime_frac(pd.to_numeric(raw_on, errors="coerce").to_numpy(dtype=np.float64), signed=signed)


def _runtime_s_to_frac(runtime: np.ndarray, interval_s: float) -> np.ndarray:
    runtime = np.where(np.isnan(runtime), 0.0, runtime)
    return np.clip(runtime / max(interval_s, 1e-6), 0.0, 1.0).astype(np.float32)


def _optional_on_frac(
    table: pd.DataFrame,
    frac_key: str,
    runtime_key: str | None,
    interval_s: float,
    *,
    signed: bool = False,
) -> np.ndarray | None:
    col = _find_column(table.columns, _COLUMN_ALIASES[frac_key])
    if col is not None:
        return _series_to_on_frac(table[col], signed=signed)
    if runtime_key is not None:
        run_col = _find_column(table.columns, _COLUMN_ALIASES[runtime_key])
        if run_col is not None:
            runtime = pd.to_numeric(table[run_col], errors="coerce").to_numpy(dtype=np.float64)
            return _runtime_s_to_frac(runtime, interval_s)
    return None


def _signed_hvac_on_frac(table: pd.DataFrame, interval_s: float) -> np.ndarray:
    """Heating minus cooling, or a generic/signed ``hvac_on_frac``.

    Cooling-only columns become negative runtime. Mixed heat+cool columns
    become reverse-cycle signed runtime in [-1, 1].
    """
    heat = _optional_on_frac(table, "heating_on_frac", "heat_runtime_s", interval_s)
    cool = _optional_on_frac(table, "cooling_on_frac", "cool_runtime_s", interval_s)
    generic = _optional_on_frac(table, "hvac_on_frac", "hvac_runtime_s", interval_s, signed=True)
    if heat is not None and cool is not None:
        return np.clip(heat - cool, -1.0, 1.0).astype(np.float32)
    if cool is not None:
        return np.clip(-cool, -1.0, 1.0).astype(np.float32)
    if heat is not None:
        return heat
    if generic is not None:
        return generic
    raise ValueError(
        "CSV needs HVAC runtime: hvac_on_frac, heating_on / cooling_on, "
        "or hvac_runtime_s / heat_runtime_s / cool_runtime_s."
    )


def timeseries_from_frame(frame: pd.DataFrame, hvac_mode: str = "auto") -> Timeseries:
    """Build a ``Timeseries`` from a thermostat/weather table.

    Required: indoor T, outdoor T, and HVAC runtime (heating, cooling, signed
    fraction, or runtime seconds). Optional: GHI, humidity, wind, setpoint,
    delivered HVAC power (`hvac_kw`). ``hvac_mode='cooling'`` negates unsigned generic runtime.
    """
    if frame.empty:
        raise ValueError("Empty timeseries table.")
    table = frame.reset_index(drop=True).copy()

    t_col = _find_column(table.columns, _COLUMN_ALIASES["t_hours"])
    ts_col = _find_column(table.columns, _COLUMN_ALIASES["timestamp"])
    if t_col is not None:
        t_hours = pd.to_numeric(table[t_col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(t_hours).any():
            raise ValueError("t_hours contains non-numeric values.")
    elif ts_col is not None:
        stamps = pd.to_datetime(table[ts_col], utc=False)
        t_hours = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
    else:
        raise ValueError("CSV needs a timestamp or t_hours column.")

    if not np.all(np.diff(t_hours) > 0):
        raise ValueError("Timeseries must be strictly increasing in time (no shuffle).")

    dt_h = float(np.median(np.diff(t_hours))) if len(t_hours) > 1 else 5.0 / 60.0
    interval_s = dt_h * 3600.0

    on_frac = _signed_hvac_on_frac(table, interval_s)

    t_in = _series(table, "t_in_c")
    t_out = _series(table, "t_out_c")
    rh_out = _as_fraction(_series(table, "rh_out", default=0.50))
    rh_in = _as_fraction(_series(table, "rh_in", default=0.40))
    # Always derive ω from the observed interval T and RH. Precomputed omega
    # columns are ignored so a plant-state export cannot leak true indoor T.
    omega_out = np.asarray(humidity_ratio(t_out, rh_out), dtype=np.float32)
    omega_in = np.asarray(humidity_ratio(t_in, rh_in), dtype=np.float32)

    data = Timeseries(
        t_hours=jnp.asarray(t_hours, dtype=jnp.float32),
        t_out_c=jnp.asarray(t_out),
        ghi_w_m2=jnp.asarray(_series(table, "ghi_w_m2", default=0.0)),
        rh_out=jnp.asarray(rh_out),
        wind_m_s=jnp.asarray(_series(table, "wind_m_s", default=0.0)),
        t_in_c=jnp.asarray(t_in),
        rh_in=jnp.asarray(rh_in),
        omega_out=jnp.asarray(omega_out),
        omega_in=jnp.asarray(omega_in),
        q_hvac_kw=jnp.asarray(_series(table, "q_hvac_kw", default=0.0)),
        hvac_on_frac=jnp.asarray(on_frac),
        q_int_kw=jnp.asarray(_series(table, "q_int_kw", default=0.0)),
        setpoint_c=jnp.asarray(_series(table, "setpoint_c", default=np.nan)),
    )
    return canonicalize_hvac(data, hvac_mode)


def load_timeseries_csv(path: str | Path, hvac_mode: str = "auto") -> Timeseries:
    """Load a thermostat/weather CSV. Time must already be chronological."""
    frame = pd.read_csv(path)
    return timeseries_from_frame(frame, hvac_mode=hvac_mode)


def timeseries_to_frame(data: Timeseries) -> pd.DataFrame:
    dt_h = float(data.t_hours[1] - data.t_hours[0]) if data.t_hours.shape[0] > 1 else 5.0 / 60.0
    u = np.asarray(data.hvac_on_frac)
    return pd.DataFrame(
        {
            "t_hours": np.asarray(data.t_hours),
            "t_out_c": np.asarray(data.t_out_c),
            "ghi_w_m2": np.asarray(data.ghi_w_m2),
            "rh_out_frac": np.asarray(data.rh_out),
            "wind_m_s": np.asarray(data.wind_m_s),
            "t_in_c": np.asarray(data.t_in_c),
            "rh_in_frac": np.asarray(data.rh_in),
            "heat_setpoint_c": np.asarray(data.setpoint_c),
            "hvac_kw": np.asarray(data.q_hvac_kw),
            "hvac_on_frac": u,
            "heating_on_frac": np.clip(u, 0.0, 1.0),
            "cooling_on_frac": np.clip(-u, 0.0, 1.0),
            "hvac_runtime_s": np.abs(u) * dt_h * 3600.0,
        }
    )


def _to_numpy_tree(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)


def _to_jax_tree(tree):
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def save_checkpoint(
    path: str | Path,
    params: ModelParams,
    meta: dict[str, Any] | None = None,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": _to_numpy_tree(params), "meta": meta or {}}
    dest.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return dest


def load_checkpoint(path: str | Path) -> tuple[ModelParams, dict[str, Any]]:
    payload = pickle.loads(Path(path).read_bytes())
    raw = payload["params"]
    params = ModelParams(
        phys=raw.phys if hasattr(raw, "phys") else raw["phys"],
        remainder_net=list(raw.remainder_net if hasattr(raw, "remainder_net") else raw["remainder_net"]),
        sigma_net=list(raw.sigma_net if hasattr(raw, "sigma_net") else raw["sigma_net"]),
        feat_mean=raw.feat_mean if hasattr(raw, "feat_mean") else raw["feat_mean"],
        feat_std=raw.feat_std if hasattr(raw, "feat_std") else raw["feat_std"],
    )
    params = _to_jax_tree(params)
    return params, dict(payload.get("meta") or {})


def estimates_json(params: ModelParams, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    building = decode_building(params.phys)
    noise = decode_noise(params.phys)
    body = {
        "C_kwh_per_k": float(building.C),
        "R_k_per_kw": float(building.R),
        "Q_rated_kw": float(building.Q_rated),
        "A_s_m2": float(building.A_s),
        "beta": float(building.beta),
        "sigma_T": float(noise.sigma_T),
        "sigma_q": float(noise.sigma_q),
        "sigma_y": float(noise.sigma_y),
        "kappa": float(noise.kappa),
    }
    if extra:
        body.update(extra)
    return body


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest
