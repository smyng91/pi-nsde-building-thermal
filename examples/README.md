# Examples

These scripts train and run inference on a thermostat/weather CSV. Indoor temperature is
the measurement, not the success metric: the score that matters is
**holdout open-loop** $T$ with frozen $C$, $R$, and (by default)
learned $Q_{\mathrm{rated}}$.

From the repo root, after `uv sync --extra dev` (see the [README](../README.md)):

```bash
uv run python examples/generate_synthetic.py
uv run python examples/train_csv.py output/synthetic_thermostat.csv
uv run python examples/infer_csv.py output/synthetic_thermostat.csv output/checkpoint.pkl --mode holdout
```

Or, in one shot:

```bash
uv run python examples/run_demo.py
```

Custom files go through the same `train_csv.py` / `infer_csv.py` entry
points. Time must already be chronological (no shuffled rows).

| Script | Role |
|---|---|
| `generate_synthetic.py` | Digital-twin CSV (not a lab trace) |
| `train_csv.py` | Two-stage ID, chronological holdout, checkpoint |
| `infer_csv.py` | Open-loop / holdout / filter diagnostic |
| `run_demo.py` | Generate → train (unknown $Q_{\mathrm{rated}}$) → holdout infer |

Default `--q-rated unknown`: the identifier sees **signed runtime**
and learns rated capacity. Heating is positive, cooling is negative.
`--hvac-mode cooling` negates unsigned generic on/off. `--q-rated known` is the
optimistic protocol that consumes a `hvac_kw` column (negative while cooling).

Outputs: `output/checkpoint.pkl`, `output/estimates.json`.

Required CSV columns (aliases accepted; see `pi_nsde_building_thermal.io`):

- indoor temperature (`t_in_c`, `indoor_temp`, …)
- outdoor temperature (`t_out_c`, `outdoor_temp`, …)
- HVAC runtime (`hvac_on_frac`, `heating_on`, `cooling_on`, or `hvac_runtime_s`)
- `timestamp` or `t_hours`

Optional: GHI, outdoor/indoor RH, wind, setpoint, `hvac_kw`.
`q_int_kw_hidden` is ignored by training even if present.
