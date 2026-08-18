#!/usr/bin/env python3
"""Publication figures and LaTeX macros: known vs unknown HVAC capacity.

Heating (winter) and cooling (summer) twins share the same plant (C, R, Q_rated)
and chronological holdout. Numbers are MAP / Laplace sd / holdout open-loop RMSE
from the fitted JAX identifier.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pi_nsde_building_thermal.io import write_json  # noqa: E402
from pi_nsde_building_thermal.synthetic import (  # noqa: E402
    SyntheticConfig,
    TRUE_PARAMS,
    generate_synthetic_building,
)
from pi_nsde_building_thermal.train import TrainConfig, identify_building  # noqa: E402
from pi_nsde_building_thermal.uq import quantify_uncertainty  # noqa: E402

PAPER = ROOT / "paper"
FIGS = PAPER / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "lines.linewidth": 1.4,
        "grid.alpha": 0.25,
    }
)

COLOR_TRUE = "0.62"
COLOR_KNOWN = "#15803d"
COLOR_UNKNOWN = "#b91c1c"
COLOR_T = "0.45"

SEASON_STEMS = {
    "heating": {
        "holdout": "fig1_holdout",
        "params": "fig2_params",
        "hvac": "fig3_hvac",
        "label": "winter heating",
        "prefix": "",
    },
    "cooling": {
        "holdout": "fig5_holdout_cool",
        "params": "fig4_params_cool",
        "hvac": "fig6_hvac_cool",
        "label": "summer cooling",
        "prefix": "Cool",
    },
}


def _tex_macros(values: dict[str, str]) -> str:
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()) + "\n"


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _rel_pct(est: float, true: float) -> str:
    return f"{100.0 * abs(est - true) / abs(true):.1f}"


def _named(uq) -> dict[str, float]:
    return {n: float(uq.laplace.mean[i]) for i, n in enumerate(uq.laplace.names)}


def _sds(uq) -> dict[str, float]:
    return {n: float(uq.laplace.sd[i]) for i, n in enumerate(uq.laplace.names)}


def fit_protocol(dataset, q_rated: str, seed: int = 1):
    cfg = TrainConfig(seed=seed, q_rated=q_rated, hvac_mode="auto")
    ident = identify_building(dataset.arrays, cfg, holdout_days=2.0, verbose=True)
    uq = quantify_uncertainty(
        ident.fit.params,
        ident.train,
        ident.fit.filter,
        cfg,
        ident.fit.n_sub,
        remainder_gate=ident.fit.remainder_gate,
        lambda_id=ident.fit.lambda_id,
    )
    return ident, uq


def write_figures(dataset, known, unknown, stems: dict[str, str], season_label: str) -> None:
    kn_ident, kn_uq = known
    un_ident, un_uq = unknown
    t = np.asarray(dataset.arrays.t_hours) / 24.0
    n_train = un_ident.split.n_train
    t_in = np.asarray(dataset.arrays.t_in_c)
    on = np.asarray(dataset.arrays.hvac_on_frac)
    q_true = np.asarray(dataset.arrays.q_hvac_kw)
    q_hat = float(un_ident.fit.estimated.Q_rated)
    split = t[n_train]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, layout="constrained")
    panels = [
        (
            axes[0],
            kn_ident,
            COLOR_KNOWN,
            rf"Known $Q_\mathrm{{hvac}}$ (holdout RMSE {kn_ident.holdout_rmse:.3f} K)",
        ),
        (
            axes[1],
            un_ident,
            COLOR_UNKNOWN,
            rf"Unknown $Q_\mathrm{{rated}}$ (holdout RMSE {un_ident.holdout_rmse:.3f} K)",
        ),
    ]
    for ax, ident, color, title in panels:
        ol = np.asarray(ident.holdout_open_loop.y_pred)
        ax.plot(t, t_in, color=COLOR_T, lw=0.85, label="Thermostat interval mean")
        ax.plot(
            t[:n_train],
            np.asarray(ident.fit.filter.t_mean),
            color="#b91c1c",
            lw=0.65,
            alpha=0.4,
            label="Train Kalman (not a metric)",
        )
        ax.plot(t[n_train:], ol, color=color, lw=1.35, label="Holdout open-loop")
        ax.axvline(split, color="0.2", ls="--", lw=0.8)
        ax.set_ylabel(r"Indoor $T$ ($^\circ$C)")
        ax.set_title(title, loc="left", fontsize=9)
        ax.legend(loc="best")
    axes[1].set_xlabel("Day")
    fig.savefig(FIGS / f"{stems['holdout']}.png")
    plt.close(fig)
    kn_ol = np.full_like(t, np.nan, dtype=float)
    un_ol = np.full_like(t, np.nan, dtype=float)
    kn_ol[n_train:] = np.asarray(kn_ident.holdout_open_loop.y_pred)
    un_ol[n_train:] = np.asarray(un_ident.holdout_open_loop.y_pred)
    np.savetxt(
        FIGS / f"{stems['holdout']}.csv",
        np.column_stack([t, t_in, kn_ol, un_ol]),
        delimiter=",",
        header="day,t_in_c,known_holdout_open_loop,unknown_holdout_open_loop",
        comments="",
    )

    kn_map, kn_sd = _named(kn_uq), _sds(kn_uq)
    un_map, un_sd = _named(un_uq), _sds(un_uq)
    true_c, true_r, true_q = TRUE_PARAMS.C, TRUE_PARAMS.R, float(TRUE_PARAMS.Q_rated)

    fig, (ax_cr, ax_q) = plt.subplots(1, 2, figsize=(7.2, 3.35), layout="constrained")
    xs = np.arange(2)
    w = 0.25
    ax_cr.bar(xs - w, [true_c, true_r], width=w, color=COLOR_TRUE, label="Plant (eval only)")
    ax_cr.bar(
        xs,
        [kn_map["C"], kn_map["R"]],
        width=w,
        color=COLOR_KNOWN,
        yerr=1.96 * np.array([kn_sd["C"], kn_sd["R"]]),
        capsize=3,
        label=r"Known $Q_\mathrm{hvac}$",
    )
    ax_cr.bar(
        xs + w,
        [un_map["C"], un_map["R"]],
        width=w,
        color=COLOR_UNKNOWN,
        yerr=1.96 * np.array([un_sd["C"], un_sd["R"]]),
        capsize=3,
        label=r"Unknown $Q_\mathrm{rated}$",
    )
    ax_cr.set_xticks(xs, [r"$C$ (kWh/K)", r"$R$ (K/kW)"])
    ax_cr.set_ylabel("MAP $\pm 1.96$ sd")
    ax_cr.legend(loc="upper right", fontsize=7)
    ax_cr.set_title(rf"(a) Envelope parameters ({season_label})", loc="left", fontsize=9)

    ax_q.bar([0], [true_q], width=0.45, color=COLOR_TRUE, label="Plant (eval only)")
    ax_q.bar(
        [1],
        [un_map["Q_rated"]],
        width=0.45,
        color=COLOR_UNKNOWN,
        yerr=1.96 * un_sd["Q_rated"],
        capsize=3,
        label=r"Unknown $Q_\mathrm{rated}$",
    )
    ax_q.set_xticks([0, 1], ["Plant", r"Unknown MAP"])
    ax_q.set_ylabel(r"$Q_\mathrm{rated}$ (kW)")
    ax_q.set_title(r"(b) Rated capacity (runtime protocol)", loc="left", fontsize=9)
    ax_q.legend(loc="upper right", fontsize=7)
    fig.savefig(FIGS / f"{stems['params']}.png")
    plt.close(fig)
    np.savetxt(
        FIGS / f"{stems['params']}.csv",
        np.array(
            [
                [true_c, kn_map["C"], kn_sd["C"], un_map["C"], un_sd["C"]],
                [true_r, kn_map["R"], kn_sd["R"], un_map["R"], un_sd["R"]],
                [true_q, np.nan, np.nan, un_map["Q_rated"], un_sd["Q_rated"]],
            ]
        ),
        delimiter=",",
        header="true,known_map,known_sd,unknown_map,unknown_sd",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    ax.plot(t, q_true, color="0.35", lw=0.85, label=r"Plant $Q_\mathrm{hvac}$ (known-kW input)")
    ax.plot(
        t,
        q_hat * on,
        color=COLOR_UNKNOWN,
        lw=0.95,
        label=r"Unknown-protocol MAP $Q_\mathrm{rated}u$",
    )
    ax.axvline(split, color="0.2", ls="--", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("kW")
    ax.legend()
    fig.savefig(FIGS / f"{stems['hvac']}.png")
    plt.close(fig)
    np.savetxt(
        FIGS / f"{stems['hvac']}.csv",
        np.column_stack([t, on, q_hat * on, q_true]),
        delimiter=",",
        header="day,on_frac,q_unknown_map,q_true",
        comments="",
    )


def _season_macros(dataset, known, unknown, prefix: str) -> dict[str, str]:
    kn_ident, kn_uq = known
    un_ident, un_uq = unknown
    kn_map, kn_sd = _named(kn_uq), _sds(kn_uq)
    un_map, un_sd = _named(un_uq), _sds(un_uq)
    true = dataset.true_params
    true_rc = float(true.C * true.R)
    true_qc = float(true.Q_rated) / float(true.C)
    kn_rc = kn_map["C"] * kn_map["R"]
    un_rc = un_map["C"] * un_map["R"]
    un_qc = un_map["Q_rated"] / un_map["C"]
    p = prefix
    return {
        f"{p}TrueC": _fmt(true.C, 2),
        f"{p}TrueR": _fmt(true.R, 2),
        f"{p}TrueQrated": _fmt(float(true.Q_rated), 2),
        f"{p}TrueAs": _fmt(true.A_s, 2),
        f"{p}TrueRC": _fmt(true_rc, 1),
        f"{p}TrueQC": _fmt(true_qc, 2),
        f"{p}Ntrain": str(un_ident.split.n_train),
        f"{p}Nhold": str(un_ident.split.n_holdout),
        f"{p}TrainDays": _fmt(un_ident.split.train_days, 0),
        f"{p}HoldDays": _fmt(un_ident.split.holdout_days, 0),
        f"{p}KnownC": _fmt(kn_map["C"], 2),
        f"{p}KnownR": _fmt(kn_map["R"], 2),
        f"{p}KnownAs": _fmt(kn_map["A_s"], 2),
        f"{p}KnownCrel": _rel_pct(kn_map["C"], true.C),
        f"{p}KnownRrel": _rel_pct(kn_map["R"], true.R),
        f"{p}KnownAsrel": _rel_pct(kn_map["A_s"], true.A_s),
        f"{p}KnownRCrel": _rel_pct(kn_rc, true_rc),
        f"{p}KnownHoldRMSE": _fmt(kn_ident.holdout_rmse, 3),
        f"{p}KnownHoldMAE": _fmt(kn_ident.holdout_mae, 3),
        f"{p}KnownTrainNLL": _fmt(kn_ident.train_nll_mean, 3),
        f"{p}KnownCsd": _fmt(kn_sd["C"], 2),
        f"{p}KnownRsd": _fmt(kn_sd["R"], 2),
        f"{p}KnownRC": _fmt(kn_rc, 1),
        f"{p}UnknownC": _fmt(un_map["C"], 2),
        f"{p}UnknownR": _fmt(un_map["R"], 2),
        f"{p}UnknownQ": _fmt(un_map["Q_rated"], 2),
        f"{p}UnknownAs": _fmt(un_map["A_s"], 2),
        f"{p}UnknownCrel": _rel_pct(un_map["C"], true.C),
        f"{p}UnknownRrel": _rel_pct(un_map["R"], true.R),
        f"{p}UnknownQrel": _rel_pct(un_map["Q_rated"], float(true.Q_rated)),
        f"{p}UnknownAsrel": _rel_pct(un_map["A_s"], true.A_s),
        f"{p}UnknownRCrel": _rel_pct(un_rc, true_rc),
        f"{p}UnknownQCrel": _rel_pct(un_qc, true_qc),
        f"{p}UnknownHoldRMSE": _fmt(un_ident.holdout_rmse, 3),
        f"{p}UnknownHoldMAE": _fmt(un_ident.holdout_mae, 3),
        f"{p}UnknownTrainNLL": _fmt(un_ident.train_nll_mean, 3),
        f"{p}UnknownCsd": _fmt(un_sd["C"], 2),
        f"{p}UnknownRsd": _fmt(un_sd["R"], 2),
        f"{p}UnknownQsd": _fmt(un_sd["Q_rated"], 2),
        f"{p}UnknownRC": _fmt(un_rc, 1),
        f"{p}UnknownQC": _fmt(un_qc, 2),
    }


def _merge_tex_macros(new_values: dict[str, str]) -> None:
    path = PAPER / "generated_numbers.tex"
    existing: dict[str, str] = {}
    if path.exists():
        for match in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", path.read_text(encoding="utf-8")):
            existing[match.group(1)] = match.group(2)
    existing.update(new_values)
    path.write_text(_tex_macros(existing), encoding="utf-8")


def _season_payload(dataset, known, unknown) -> dict:
    kn_ident, kn_uq = known
    un_ident, un_uq = unknown
    kn_map, kn_sd = _named(kn_uq), _sds(kn_uq)
    un_map, un_sd = _named(un_uq), _sds(un_uq)
    true = dataset.true_params
    kn_est = kn_ident.fit.estimated
    un_est = un_ident.fit.estimated
    return {
        "true": {
            "C": float(true.C),
            "R": float(true.R),
            "Q_rated": float(true.Q_rated),
            "A_s": float(true.A_s),
            "RC_h": float(true.C * true.R),
            "Q_over_C": float(true.Q_rated) / float(true.C),
        },
        "known_qhvac": {
            "C": kn_map["C"],
            "R": kn_map["R"],
            "A_s": kn_map["A_s"],
            "sd": kn_sd,
            "holdout_rmse": kn_ident.holdout_rmse,
            "holdout_mae": kn_ident.holdout_mae,
            "train_nll_mean": kn_ident.train_nll_mean,
            "RC_h": kn_map["C"] * kn_map["R"],
            "estimated": {
                "C": float(kn_est.C),
                "R": float(kn_est.R),
                "A_s": float(kn_est.A_s),
                "beta": float(kn_est.beta),
            },
        },
        "unknown_qrated": {
            "C": un_map["C"],
            "R": un_map["R"],
            "Q_rated": un_map["Q_rated"],
            "A_s": un_map["A_s"],
            "sd": un_sd,
            "holdout_rmse": un_ident.holdout_rmse,
            "holdout_mae": un_ident.holdout_mae,
            "train_nll_mean": un_ident.train_nll_mean,
            "RC_h": un_map["C"] * un_map["R"],
            "Q_over_C": un_map["Q_rated"] / un_map["C"],
            "estimated": {
                "C": float(un_est.C),
                "R": float(un_est.R),
                "Q_rated": float(un_est.Q_rated),
                "A_s": float(un_est.A_s),
                "beta": float(un_est.beta),
            },
        },
    }


def run_season(hvac_mode: str) -> dict:
    meta = SEASON_STEMS[hvac_mode]
    print(f"=== Paired HVAC-capacity study ({meta['label']}) ===")
    dataset = generate_synthetic_building(SyntheticConfig(days=7, seed=0, hvac_mode=hvac_mode))
    print("\n----- Protocol: known delivered kW -----\n")
    known = fit_protocol(dataset, "known")
    print("\n----- Protocol: unknown Q_rated (runtime only) -----\n")
    unknown = fit_protocol(dataset, "unknown")
    write_figures(dataset, known, unknown, meta, meta["label"])
    macros = _season_macros(dataset, known, unknown, meta["prefix"])
    _merge_tex_macros(macros)
    payload = _season_payload(dataset, known, unknown)
    kn, un = payload["known_qhvac"], payload["unknown_qrated"]
    print(f"\n=== {meta['label']} comparison ===")
    print(
        f"  known:    C {kn['C']:.3f}  R {kn['R']:.3f}  "
        f"holdout RMSE {kn['holdout_rmse']:.3f} K"
    )
    print(
        f"  unknown:  C {un['C']:.3f}  R {un['R']:.3f}  Q {un['Q_rated']:.3f}  "
        f"holdout RMSE {un['holdout_rmse']:.3f} K"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        choices=("heating", "cooling", "both"),
        default="both",
        help="Which digital-twin season(s) to fit. Default both.",
    )
    args = parser.parse_args()
    seasons = ("heating", "cooling") if args.season == "both" else (args.season,)
    manifest: dict = {}
    if (FIGS / "manifest.json").exists():
        import json

        loaded = json.loads((FIGS / "manifest.json").read_text(encoding="utf-8"))
        if "known_qhvac" in loaded and "heating" not in loaded:
            manifest = {"heating": loaded}
        else:
            manifest = loaded
    for season in seasons:
        manifest[season] = run_season(season)
    write_json(FIGS / "manifest.json", manifest)
    print("Wrote paper/figures and paper/generated_numbers.tex")


if __name__ == "__main__":
    main()
