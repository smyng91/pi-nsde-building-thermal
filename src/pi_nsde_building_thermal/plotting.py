"""Figures for the pi-nsde-building-thermal identification protocol (not Kalman T-tracking)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pi_nsde_building_thermal.physics import BuildingParams
from pi_nsde_building_thermal.synthetic import Timeseries
from pi_nsde_building_thermal.train import IdentificationResult
from pi_nsde_building_thermal.uq import UncertaintyReport


def plot_example(
    data: Timeseries,
    ident: IdentificationResult,
    uq: UncertaintyReport,
    true_params: BuildingParams,
    path: str | Path,
) -> None:
    t = np.asarray(data.t_hours) / 24.0
    n_train = ident.split.n_train
    t_hold = t[n_train:]
    split_day = t[n_train] if n_train < t.size else t[-1]
    unknown = ident.fit.q_rated == "unknown"
    q_hat = float(ident.fit.estimated.Q_rated)
    on_frac = np.asarray(data.hvac_on_frac)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, np.asarray(data.t_out_c), color="#1f77b4", lw=1.1, label="Outdoor T")
    ax.set_ylabel("Outdoor temperature [°C]")
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, np.asarray(data.ghi_w_m2), color="#f4c430", alpha=0.35, label="GHI")
    ax2.set_ylabel("GHI [W/m²]")
    ax.axvline(split_day, color="0.3", ls="--", lw=0.9, label="Holdout start")
    ax.set_title("Ambient conditions")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, np.asarray(data.t_in_c), color="0.45", lw=0.9, label="Thermostat interval mean")
    ax.plot(
        t[:n_train],
        np.asarray(ident.fit.filter.t_mean),
        color="#d62728",
        lw=0.8,
        alpha=0.55,
        label="Train Kalman (uses train T; not a metric)",
    )
    ax.plot(
        t_hold,
        np.asarray(ident.holdout_open_loop.y_pred),
        color="#2ca02c",
        lw=1.25,
        label="Holdout open-loop T (primary T metric)",
    )
    ax.plot(t, np.asarray(data.setpoint_c), color="#9467bd", lw=0.8, ls=":", label="Setpoint")
    ax.axvline(split_day, color="0.3", ls="--", lw=0.9)
    ax.set_ylabel("Indoor temperature [°C]")
    ax.set_title(
        f"Holdout open-loop RMSE={ident.holdout_rmse:.2f} K  MAE={ident.holdout_mae:.2f} K"
    )
    ax.legend(loc="best", fontsize=7)

    ax = axes[1, 0]
    ax2 = None
    if unknown:
        ax.plot(t, q_hat * on_frac, color="#c44e52", lw=0.9, label=f"MAP Q_rated×on_frac ({q_hat:.2f} kW)")
        ax.plot(
            t,
            np.asarray(data.q_hvac_kw),
            color="#c44e52",
            lw=0.7,
            ls="--",
            alpha=0.45,
            label="True delivered kW (eval only; unused in fit)",
        )
        ax2 = ax.twinx()
        ax2.plot(t, on_frac, color="#ff7f0e", lw=0.7, alpha=0.7, label="signed runtime (observed)")
        ax2.set_ylabel("HVAC signed runtime")
        ymin = -1.15 if float(np.min(on_frac)) < -1e-3 else -0.05
        ax2.set_ylim(ymin, 1.15)
        ax.set_title("Observed runtime; capacity identified (true kW unused)")
    else:
        ax.plot(t, np.asarray(data.q_hvac_kw), color="#c44e52", lw=0.9, label="Q_hvac (known kW)")
        ax.set_title("Known HVAC vs latent occupancy (hidden Q_int not used in fit)")
    ax.plot(t, np.asarray(data.q_int_kw), color="0.45", lw=0.8, label="Q_int true (eval only)")
    q_tr = np.asarray(uq.q_mean)
    q_sd = np.asarray(uq.q_sd_state)
    ax.fill_between(
        t[:n_train],
        q_tr - 2 * q_sd,
        q_tr + 2 * q_sd,
        color="#1f77b4",
        alpha=0.2,
        label="Train filtered Q_int ±2σ",
    )
    ax.plot(t[:n_train], q_tr, color="#1f77b4", lw=1.0, label="Train filtered Q_int")
    ax.plot(
        t_hold,
        np.asarray(ident.holdout_open_loop.q_mean),
        color="#17becf",
        lw=1.1,
        label="Holdout open-loop Q_int mean",
    )
    ax.axvline(split_day, color="0.3", ls="--", lw=0.9)
    ax.set_xlabel("Day")
    ax.set_ylabel("kW")
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, fontsize=6, loc="upper right")

    ax = axes[1, 1]
    names = list(uq.laplace.names[:3] if unknown else uq.laplace.names[:2])
    if unknown:
        true = np.array([true_params.C, true_params.R, float(true_params.Q_rated)])
    else:
        true = np.array([true_params.C, true_params.R])
    est = np.array([float(uq.laplace.mean[i]) for i in range(len(names))])
    sd = np.array([float(uq.laplace.sd[i]) for i in range(len(names))])
    xs = np.arange(len(names))
    ax.bar(xs - 0.15, true, width=0.3, color="0.65", label="True (eval only)")
    ax.bar(
        xs + 0.15,
        est,
        width=0.3,
        color="#d62728",
        yerr=1.96 * sd,
        capsize=4,
        label="Train MAP ± 1.96 sd (Laplace)",
    )
    ax.set_xticks(xs, names)
    ax.set_ylabel("C [kWh/K], R [K/kW]" + (", Q_rated [kW]" if unknown else ""))
    ax.set_title("Train only (joint Laplace, sum NLL)")
    ax.legend(fontsize=8)

    mode = "unknown Q_rated (on/off only)" if unknown else "known HVAC kW"
    fig.suptitle(f"pi-nsde-building-thermal identification — chronological holdout, two-stage, {mode}", fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
