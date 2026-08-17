#!/usr/bin/env python3
"""Open-loop inference on a custom CSV with a saved checkpoint.

Uses weather + HVAC on/off (× estimated Q_rated in unknown mode). Does not
update from indoor T except as the initial condition (or last train filter
state in ``--mode holdout``). ``--mode filter`` is a diagnostic that sees T;
it is not the identification metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _paths import out_dir

from pinn_building.io import estimates_json, load_checkpoint, load_timeseries_csv, timeseries_to_frame, write_json
from pinn_building.synthetic import chronological_split
from pinn_building.train import filter_from_params, open_loop_from_params, temperature_rmse_mae


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--mode", choices=("open-loop", "holdout", "filter"), default="open-loop")
    p.add_argument("--holdout-days", type=float, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    dest_dir = out_dir()
    arrays = load_timeseries_csv(args.csv)
    params, meta = load_checkpoint(args.checkpoint)
    q_rated = meta.get("q_rated", "unknown")
    n_sub = int(meta.get("n_sub", 5))
    gate = float(meta.get("remainder_gate", 1.0))

    if args.mode == "holdout":
        train, holdout, split = chronological_split(arrays, holdout_days=args.holdout_days)
        filt = filter_from_params(params, train, n_sub, remainder_gate=gate, q_rated_mode=q_rated)
        ol = open_loop_from_params(
            params,
            holdout,
            n_sub,
            t0=float(filt.t_mean[-1]),
            q0=float(filt.q_mean[-1]),
            remainder_gate=gate,
            q_rated_mode=q_rated,
        )
        rmse, mae = temperature_rmse_mae(ol.y_pred, holdout.t_in_c)
        out_frame = timeseries_to_frame(holdout)
        out_frame["t_open_loop_c"] = ol.y_pred
        print(
            f"Holdout open-loop [{split.n_train}, {split.n_total}): "
            f"RMSE={rmse:.4f} K  MAE={mae:.4f} K"
        )
        extra = {"mode": "holdout", "rmse_k": rmse, "mae_k": mae, "split": split.scheme}
    elif args.mode == "filter":
        filt = filter_from_params(params, arrays, n_sub, remainder_gate=gate, q_rated_mode=q_rated)
        out_frame = timeseries_to_frame(arrays)
        out_frame["t_filter_c"] = filt.t_mean
        print("Filter diagnostic uses indoor T. This is not the identification metric.")
        extra = {"mode": "filter", "train_nll": float(filt.nll)}
    else:
        ol = open_loop_from_params(
            params,
            arrays,
            n_sub,
            t0=float(arrays.t_in_c[0]),
            q0=0.7,
            remainder_gate=gate,
            q_rated_mode=q_rated,
        )
        rmse, mae = temperature_rmse_mae(ol.y_pred, arrays.t_in_c)
        out_frame = timeseries_to_frame(arrays)
        out_frame["t_open_loop_c"] = ol.y_pred
        print(
            f"Open-loop from first indoor T (weather + on/off only): "
            f"RMSE={rmse:.4f} K  MAE={mae:.4f} K vs the CSV T column"
        )
        extra = {"mode": "open-loop", "rmse_k": rmse, "mae_k": mae}

    dest = args.output or (dest_dir / "inference.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(dest, index=False)
    write_json(dest.with_suffix(".json"), estimates_json(params, extra))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
