#!/usr/bin/env python3
"""Write a synthetic thermostat/weather CSV for the training examples.

This is a digital twin, not a laboratory trace. HVAC on/off is observed;
plant HVAC power is written for optional metered-Q_hvac runs and must not be
required by ``train_csv.py --q-rated unknown``.
"""

from __future__ import annotations

import argparse

from _paths import out_dir

from pi_nsde_building_thermal.io import timeseries_to_frame
from pi_nsde_building_thermal.synthetic import SyntheticConfig, generate_synthetic_building


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="CSV path (default output/synthetic_thermostat.csv)")
    p.add_argument(
        "--hvac-mode",
        choices=("heating", "cooling"),
        default="heating",
        help="heating: winter twin. cooling: summer twin with signed (negative) runtime.",
    )
    args = p.parse_args(argv)

    dataset = generate_synthetic_building(
        SyntheticConfig(days=args.days, seed=args.seed, hvac_mode=args.hvac_mode)
    )
    dest = out_dir() / "synthetic_thermostat.csv" if args.output is None else args.output
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame = timeseries_to_frame(dataset.arrays)
    if "timestamp" in dataset.frame.columns:
        frame.insert(0, "timestamp", dataset.frame["timestamp"].to_numpy())
    frame.to_csv(dest, index=False)
    print(f"Wrote {len(frame)} rows to {dest}")
    print("Required for unknown-Q_rated training: t_in_c, t_out_c, and HVAC runtime "
          "(hvac_on_frac, heating_on / cooling_on, or runtime seconds).")
    print("Do not pass q_int_kw_hidden into the identifier; it is not in this export.")


if __name__ == "__main__":
    main()
