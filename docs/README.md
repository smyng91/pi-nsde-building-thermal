# pi-nsde-building-thermal documentation

This folder documents the **physics-informed neural SDE** used to identify a house’s effective thermal capacity $C$ and envelope resistance $R$ from weather and smart-thermostat time series.

The same pages are published on the [GitHub wiki](https://github.com/smyng91/pi-nsde-building-thermal/wiki).

| Page | Contents |
| --- | --- |
| [Mathematical models](mathematical-models.md) | Lumped energy balance, SDE, interval-average observation, Euler–Maruyama / Kalman, HVAC runtime vs delivered kW, identifiability |
| [Framework](framework.md) | Package layout, data I/O, two-stage MAP, holdout open-loop metric, Laplace UQ, CSV examples |

## What the identifier is

The identifier is a gray-box **stochastic differential equation** in JAX. Indoor temperature is the **measurement**, not the score. HVAC **runtime** is observed and **signed** (heating positive, cooling negative); rated capacity $`Q_{\mathrm{rated}}`$ is optional. The default protocol is `--q-rated unknown`.

A Kalman mean that tracks the thermostat series is **not** a success metric. The dynamics check is a chronological **holdout open-loop** rollout that sees weather and runtime (or metered kW), not a filter overlay of the same $T$.

## Units

Time is in hours so that $`C\,\mathrm{d}T/\mathrm{d}t`$ has units of power:

| Quantity | Unit |
| --- | --- |
| $C$ | $`\mathrm{kWh\,K}^{-1}`$ |
| $R$ | $`\mathrm{K\,kW}^{-1}`$ ($`UA = 1/R`$) |
| Heat fluxes | $`\mathrm{kW}`$ |
| $`Q_{\mathrm{rated}}`$ | $`\mathrm{kW}`$ |
| $u$ | signed runtime in $`[-1,1]`$ (heat +, cool −) |

## Code map

| Module | Role |
| --- | --- |
| `pi_nsde_building_thermal.physics` | Deterministic 1R1C-G energy balance |
| `pi_nsde_building_thermal.sde` | SDE, interval-average Kalman, open-loop rollout |
| `pi_nsde_building_thermal.model` | Softplus parameters, neural remainder, penalties |
| `pi_nsde_building_thermal.train` | Two-stage MAP, chronological split |
| `pi_nsde_building_thermal.uq` | Train-only Laplace Hessian |
| `pi_nsde_building_thermal.io` | CSV aliases and checkpoints |
| `pi_nsde_building_thermal.synthetic` | Digital-twin plant (evaluation only for true $C$, $R$) |

## Install and run

[uv](https://docs.astral.sh/uv/), Python 3.10+. Same commands on Windows, macOS, and Linux:

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m pi_nsde_building_thermal.example --q-rated unknown
```

CSV examples: [examples/README.md](../examples/README.md). Full setup: [README](../README.md).
