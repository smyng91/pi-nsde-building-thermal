# Examples

Training and inference on a thermostat/weather CSV. Indoor temperature is
the measurement, not the success metric: the score that matters is
**holdout open-loop** $T$ with frozen $C$, $R$, and (default)
learned $`Q_{\mathrm{rated}}`$.

```bash
.\.venv\Scripts\python.exe examples/generate_synthetic.py
.\.venv\Scripts\python.exe examples/train_csv.py output/synthetic_thermostat.csv
.\.venv\Scripts\python.exe examples/infer_csv.py output/synthetic_thermostat.csv output/checkpoint.pkl --mode holdout
```

Or one shot:

```bash
.\.venv\Scripts\python.exe examples/run_demo.py
```

Custom files go through the same `train_csv.py` / `infer_csv.py` entry
points. Time must already be chronological (no shuffled rows).

| Script | Role |
|---|---|
| `generate_synthetic.py` | Digital-twin CSV (not a lab trace) |
| `train_csv.py` | Two-stage ID, chronological holdout, checkpoint |
| `infer_csv.py` | Open-loop / holdout / filter diagnostic |
| `run_demo.py` | Generate → train (unknown $`Q_{\mathrm{rated}}`$) → holdout infer |

Default `--q-rated unknown`: the identifier sees **on/off / runtime
fraction only** and learns rated capacity. `--q-rated known` is the
optimistic protocol that consumes a `hvac_kw` column.

Outputs: `output/checkpoint.pkl`, `output/estimates.json`.

Required CSV columns (aliases accepted; see `pinn_building.io`):

- indoor temperature (`t_in_c`, `indoor_temp`, …)
- outdoor temperature (`t_out_c`, `outdoor_temp`, …)
- HVAC on/off (`hvac_on_frac`, `hvac_on`, or `hvac_runtime_s`)
- `timestamp` or `t_hours`

Optional: GHI, outdoor/indoor RH, wind, setpoint, `hvac_kw`.
`q_int_kw_hidden` is ignored by training even if present.
