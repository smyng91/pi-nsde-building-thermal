#!/usr/bin/env python3
"""Generate a synthetic CSV, fit unknown Q_rated, and run holdout inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from _paths import out_dir
from generate_synthetic import main as generate_main
from infer_csv import main as infer_main
from train_csv import main as train_main


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-a", type=int, default=1800)
    p.add_argument("--steps-b-freeze", type=int, default=300)
    p.add_argument("--steps-b-joint", type=int, default=1400)
    p.add_argument(
        "--hvac-mode",
        choices=("heating", "cooling"),
        default="heating",
    )
    args = p.parse_args(argv)

    dest = out_dir()
    csv = dest / "synthetic_thermostat.csv"
    generate_main(
        [
            "--days",
            str(args.days),
            "--seed",
            str(args.seed),
            "--output",
            str(csv),
            "--hvac-mode",
            args.hvac_mode,
        ]
    )
    train_main(
        [
            str(csv),
            "--q-rated",
            "unknown",
            "--hvac-mode",
            "auto",
            "--steps-a",
            str(args.steps_a),
            "--steps-b-freeze",
            str(args.steps_b_freeze),
            "--steps-b-joint",
            str(args.steps_b_joint),
            "--output-dir",
            str(dest),
        ]
    )
    infer_main(
        [
            str(csv),
            str(dest / "checkpoint.pkl"),
            "--mode",
            "holdout",
            "--output",
            str(dest / "holdout_open_loop.csv"),
        ]
    )


if __name__ == "__main__":
    main()
