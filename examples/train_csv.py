#!/usr/bin/env python3
"""Fit C, R, and (by default) Q_rated from a custom thermostat CSV.

Chronological holdout. HVAC on/off is observed. Default ``--q-rated unknown``
does not read delivered kilowatts. Indoor T is the measurement, not the score:
the printed T metric is holdout open-loop rollout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _paths import out_dir

from pinn_building.io import estimates_json, load_timeseries_csv, save_checkpoint, write_json
from pinn_building.train import TrainConfig, identify_building
from pinn_building.uq import quantify_uncertainty


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Chronological thermostat/weather CSV")
    p.add_argument("--q-rated", choices=("unknown", "known"), default="unknown")
    p.add_argument("--holdout-days", type=float, default=None)
    p.add_argument("--steps-a", type=int, default=1800)
    p.add_argument("--steps-b-freeze", type=int, default=300)
    p.add_argument("--steps-b-joint", type=int, default=1400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    dest = args.output_dir or out_dir()
    dest.mkdir(parents=True, exist_ok=True)

    arrays = load_timeseries_csv(args.csv)
    cfg = TrainConfig(
        stage_a_steps=args.steps_a,
        stage_b_freeze_cr_steps=args.steps_b_freeze,
        stage_b_joint_steps=args.steps_b_joint,
        seed=args.seed,
        q_rated=args.q_rated,
    )
    ident = identify_building(arrays, cfg, holdout_days=args.holdout_days, verbose=True)
    uq = quantify_uncertainty(
        ident.fit.params,
        ident.train,
        ident.fit.filter,
        cfg,
        ident.fit.n_sub,
        remainder_gate=ident.fit.remainder_gate,
        lambda_id=ident.fit.lambda_id,
    )

    names = list(uq.laplace.names)
    print("\nTrain MAP (Laplace sd from train sum-NLL Hessian; nets frozen at MAP)")
    print(f"{'parameter':<12} {'MAP':>10} {'sd':>10} {'95% CI':>22}")
    sd_map, ci_map, est_map = {}, {}, {}
    for i, name in enumerate(names):
        est = float(uq.laplace.mean[i])
        sd = float(uq.laplace.sd[i])
        rel = min(sd / max(abs(est), 1e-6), 3.0)
        lo = est * math.exp(-1.96 * rel)
        hi = est * math.exp(1.96 * rel)
        print(f"{name:<12} {est:10.3f} {sd:10.3f} [{lo:8.3f}, {hi:8.3f}]")
        est_map[name] = est
        sd_map[name] = sd
        ci_map[name] = [lo, hi]

    print(f"\nHoldout open-loop T RMSE={ident.holdout_rmse:.4f} K  MAE={ident.holdout_mae:.4f} K")
    print("Do not read in-sample Kalman T overlay as identification accuracy.")

    ckpt = dest / "checkpoint.pkl"
    save_checkpoint(
        ckpt,
        ident.fit.params,
        {
            "q_rated": ident.fit.q_rated,
            "n_sub": ident.fit.n_sub,
            "remainder_gate": ident.fit.remainder_gate,
            "lambda_id": ident.fit.lambda_id,
            "holdout_days": ident.split.holdout_days,
            "n_train": ident.split.n_train,
            "source_csv": str(Path(args.csv).resolve()),
        },
    )
    payload = estimates_json(
        ident.fit.params,
        {
            "q_rated_mode": ident.fit.q_rated,
            "estimated": est_map,
            "sd": sd_map,
            "ci95": ci_map,
            "holdout_open_loop": {"rmse_k": ident.holdout_rmse, "mae_k": ident.holdout_mae},
            "split": ident.split.scheme,
            "n_train": ident.split.n_train,
            "n_holdout": ident.split.n_holdout,
            "checkpoint": str(ckpt),
        },
    )
    write_json(dest / "estimates.json", payload)
    print(f"Wrote {ckpt} and {dest / 'estimates.json'}")


if __name__ == "__main__":
    main()
