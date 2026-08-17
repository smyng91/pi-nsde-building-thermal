"""Generate synthetic thermostat data, two-stage pi-nsde-building-thermal ID, holdout open-loop T.

Indoor T is the measurement. Overlaying a Kalman filter on the thermostat series
is not a success metric. Primary T metric: chronological holdout open-loop
rollout with frozen MAP C, R, Q_rated (unknown mode) and train-fit remainder / μ_q,
using holdout weather and HVAC on/off only (estimated Q_rated × signed u).

Default ``--q-rated unknown``: the identifier sees interval runtime fraction, not
delivered kW. ``--q-rated known`` is the older optimistic metered-kW protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from pi_nsde_building_thermal.plotting import plot_example
from pi_nsde_building_thermal.synthetic import SyntheticConfig, generate_synthetic_building
from pi_nsde_building_thermal.train import TrainConfig, identify_building
from pi_nsde_building_thermal.uq import UQ_METHOD, quantify_uncertainty

# Last paired known-delivered-kW run (same 7d / last-2d / two-stage protocol
# as scripts/generate_paper_figures.py). Optimistic: identifier was given
# plant q_hvac_kw = 9 kW × signed runtime.
KNOWN_KW_REFERENCE = {
    "note": (
        "Optimistic known-delivered-kW protocol (same twin/split/stages as "
        "the unknown-Q_rated run; identifier saw plant q_hvac_kw)."
    ),
    "relative_error": {"C": 0.06581969010202508, "R": 0.017583343717786977},
    "holdout_open_loop_rmse_k": 0.20266304910182953,
    "holdout_open_loop_mae_k": 0.17341279983520508,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=7, help="Single contiguous digital-twin series.")
    p.add_argument(
        "--holdout-days",
        type=float,
        default=2.0,
        help="Last N days held out (chronological). Default last 2 of 7.",
    )
    p.add_argument("--steps-a", type=int, default=1800, help="Stage A steps (remainder frozen at 0).")
    p.add_argument(
        "--steps-b-freeze",
        type=int,
        default=300,
        help="Stage B steps with remainder on and C,R (and Q_rated) frozen.",
    )
    p.add_argument("--steps-b-joint", type=int, default=1400, help="Stage B joint fine-tune steps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--q-rated",
        choices=("known", "unknown", "both"),
        default="unknown",
        help="known: metered kW (optimistic). unknown: on/off only, learn Q_rated. both: run unknown then known.",
    )
    p.add_argument(
        "--hvac-mode",
        choices=("heating", "cooling"),
        default="heating",
        help="Digital-twin HVAC: heating (winter) or cooling (summer, signed runtime).",
    )
    return p.parse_args(argv)


def _true_map(dataset) -> dict[str, float]:
    true = dataset.true_params
    return {
        "C": true.C,
        "R": true.R,
        "Q_rated": float(dataset.config.heating_capacity_kw),
        "A_s": true.A_s,
        "beta": true.beta,
        "sigma_T": float(dataset.true_noise.sigma_T),
        "sigma_q": float(dataset.true_noise.sigma_q),
        "sigma_y": float(dataset.true_noise.sigma_y),
        "kappa": float(dataset.true_noise.kappa),
    }


def _fit_and_summarize(dataset, train_cfg: TrainConfig, holdout_days: float, verbose: bool = True) -> dict:
    ident = identify_building(
        dataset.arrays,
        train_cfg,
        holdout_days=holdout_days,
        verbose=verbose,
    )
    uq = quantify_uncertainty(
        ident.fit.params,
        ident.train,
        ident.fit.filter,
        train_cfg,
        ident.fit.n_sub,
        remainder_gate=ident.fit.remainder_gate,
        lambda_id=ident.fit.lambda_id,
    )
    true_map = _true_map(dataset)
    names = list(uq.laplace.names)
    unknown = train_cfg.q_rated == "unknown"
    if verbose:
        print("\nIdentified parameters - train Laplace MAP +/- sd (joint Hessian of sum NLL + penalties)")
        print("Fourier mu_q is joint in the Hessian; neural remainder weights are MAP only.")
        print("C, R" + (", Q_rated" if unknown else "") + " intervals use train likelihood only. Holdout was not used to fit or for UQ.")
        if unknown:
            print("Q_rated init is 6 kW (not plant 9 kW). Identifier never reads delivered q_hvac_kw.")
        print(f"{'parameter':<12} {'true':>10} {'MAP':>10} {'sd':>10} {'95% CI':>22} {'rel err':>10}")

    estimates = {}
    sd_map = {}
    ci_map = {}
    rel_map = {}
    for i, name in enumerate(names):
        tval = true_map[name]
        est = float(uq.laplace.mean[i])
        sd = float(uq.laplace.sd[i])
        rel = min(sd / max(abs(est), 1e-6), 3.0)
        lo = est * math.exp(-1.96 * rel)
        hi = est * math.exp(1.96 * rel)
        err = abs(est - tval) / abs(tval)
        if verbose:
            print(f"{name:<12} {tval:10.3f} {est:10.3f} {sd:10.3f} [{lo:8.3f}, {hi:8.3f}] {err:9.1%}")
        estimates[name] = est
        sd_map[name] = sd
        ci_map[name] = [lo, hi]
        rel_map[name] = err

    hist = ident.fit.history
    stage_final = {}
    if hist["stage"]:
        for stage in ("A", "B1", "B2"):
            idxs = [i for i, s in enumerate(hist["stage"]) if s == stage]
            if idxs:
                j = idxs[-1]
                stage_final[stage] = {
                    "C": hist["C"][j],
                    "R": hist["R"][j],
                    "Q_rated": hist["Q_rated"][j],
                    "nll": hist["nll"][j],
                    "remainder_rms": hist["remainder_rms"][j],
                }

    if verbose:
        print("\nHoldout open-loop indoor T (primary T metric)")
        print(f"  RMSE = {ident.holdout_rmse:.4f} K")
        print(f"  MAE  = {ident.holdout_mae:.4f} K")
        print(f"Secondary train mean Kalman NLL = {ident.train_nll_mean:.4f} (filter uses train T).")

    block = {
        "q_rated_mode": train_cfg.q_rated,
        "hvac_observation": "hvac_on_frac" if unknown else "q_hvac_kw",
        "q_hvac_kw_used_in_training": not unknown,
        "protocol": {
            "split": ident.split.scheme,
            "n_train": ident.split.n_train,
            "n_holdout": ident.split.n_holdout,
            "train_days": ident.split.train_days,
            "holdout_days": ident.split.holdout_days,
            "primary_t_metric": "holdout_open_loop_T_vs_thermostat",
            "kalman_t_overlay_is_success_metric": False,
            "hidden_q_int_used_in_training": False,
            "q_hvac_kw_used_in_training": not unknown,
            "holdout_open_loop_hvac": (
                "estimated_Q_rated_times_holdout_on_frac"
                if unknown
                else "holdout_q_hvac_kw"
            ),
            "stages": {
                "A": (
                    "remainder frozen at 0; fit C,R"
                    + (",Q_rated" if unknown else "")
                    + ",A_s,beta,noise,Fourier mu_q on train"
                ),
                "B1": "remainder on, C, R"
                + (", Q_rated" if unknown else "")
                + " frozen, smaller LR, stronger identifiability penalty",
                "B2": "joint fine-tune with strong remainder penalty",
            },
        },
        "true": {k: true_map[k] for k in names},
        "estimated": estimates,
        "sd": sd_map,
        "ci95": ci_map,
        "relative_error": rel_map,
        "holdout_open_loop": {"rmse_k": ident.holdout_rmse, "mae_k": ident.holdout_mae},
        "train_kalman_nll_mean": ident.train_nll_mean,
        "train_kalman_nll_sum": ident.train_nll_sum,
        "uq_method": UQ_METHOD,
        "uq_n_obs_train": int(uq.laplace.n_obs),
        "stage_final": stage_final,
    }
    return {"ident": ident, "uq": uq, "block": block}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    modes = ("unknown", "known") if args.q_rated == "both" else (args.q_rated,)

    print("=== Evaluation protocol ===")
    print("Chronological train | holdout (no shuffle, no random rows, no future-in-past features).")
    print("True C, R, Q_rated and hidden Q_int are evaluation-only on this synthetic run.")
    print("Hidden q_int_kw is not a model feature, Kalman input, or loss term.")
    print("HVAC on/off is observed; it is not a switching latent mode.")
    if "unknown" in modes:
        print("Unknown Q_rated: identifier sees hvac_on_frac only, not delivered q_hvac_kw / true 9 kW.")
        print("Holdout open-loop uses estimated Q_rated × holdout on/off + weather.")
    if "known" in modes:
        print("Known Q_rated: optimistic metered kW (plant q_hvac_kw).")
    print("Primary T metric: holdout open-loop (not in-sample Kalman T RMSE).")
    print()

    dataset = generate_synthetic_building(
        SyntheticConfig(days=args.days, seed=args.seed, hvac_mode=args.hvac_mode)
    )
    results = {}
    plot_ident = None
    plot_uq = None
    for mode in modes:
        print(f"\n======== q-rated = {mode} ========")
        train_cfg = TrainConfig(
            stage_a_steps=args.steps_a,
            stage_b_freeze_cr_steps=args.steps_b_freeze,
            stage_b_joint_steps=args.steps_b_joint,
            seed=args.seed + 1,
            q_rated=mode,
            hvac_mode="cooling" if args.hvac_mode == "cooling" else "auto",
        )
        packed = _fit_and_summarize(dataset, train_cfg, args.holdout_days, verbose=True)
        results[mode] = packed["block"]
        if plot_ident is None or mode == "unknown":
            plot_ident = packed["ident"]
            plot_uq = packed["uq"]

    ident = plot_ident
    uq = plot_uq
    primary_mode = "unknown" if "unknown" in results else args.q_rated
    primary = results[primary_mode]

    frame = dataset.frame.copy()
    frame["split"] = ["train"] * ident.split.n_train + ["holdout"] * ident.split.n_holdout
    frame.to_csv(out / "synthetic_timeseries.csv", index=False)

    summary = dict(primary)
    summary["q_rated_runs"] = {m: results[m] for m in results}
    if "unknown" in results:
        unk = results["unknown"]["relative_error"]
        summary["known_kw_reference"] = KNOWN_KW_REFERENCE
        if "known" in results:
            summary["known_kw_reference"] = {
                "note": "This run's --q-rated known comparison (same data/split/steps).",
                "relative_error": {
                    "C": results["known"]["relative_error"]["C"],
                    "R": results["known"]["relative_error"]["R"],
                },
                "holdout_open_loop_rmse_k": results["known"]["holdout_open_loop"]["rmse_k"],
                "holdout_open_loop_mae_k": results["known"]["holdout_open_loop"]["mae_k"],
            }
        print("\n--- C / R / Q_rated vs known-delivered-kW reference ---")
        print(
            f"  unknown:  C {unk.get('C', float('nan')):.1%}  "
            f"R {unk.get('R', float('nan')):.1%}  "
            f"Q_rated {unk.get('Q_rated', float('nan')):.1%}  "
            f"holdout RMSE {results['unknown']['holdout_open_loop']['rmse_k']:.3f} K"
        )
        ref = summary["known_kw_reference"]["relative_error"]
        print(
            f"  known kW: C {ref['C']:.1%}  R {ref['R']:.1%}  "
            f"holdout RMSE {summary['known_kw_reference']['holdout_open_loop_rmse_k']:.3f} K"
        )

    (out / "parameter_estimates.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(ident.fit.history).to_csv(out / "training_history.csv", index=False)
    plot_example(
        dataset.arrays,
        ident,
        uq,
        dataset.true_params,
        out / "identification.png",
    )
    print(f"\nWrote CSV, JSON, and figure to {out.resolve()}")
    print("UQ: joint Laplace on train sum-NLL MAP; neural remainder weights frozen at MAP.")
    print("Do not read the train Kalman T overlay as identification accuracy.")


if __name__ == "__main__":
    main()
