# PI-NSDE building thermal documentation

This folder documents the **physics-informed neural SDE** used to identify a house’s effective thermal capacity $C$ and envelope resistance $R$ from weather and smart-thermostat timeseries.

The same pages are published on the [GitHub wiki](https://github.com/smyng91/pi-nde-building-thermal/wiki).

| Page | Contents |
| --- | --- |
| [Mathematical models](mathematical-models.md) | Lumped energy balance, SDE, interval-average observation, Euler–Maruyama / Kalman, HVAC runtime vs delivered kW, identifiability |
| [Framework](framework.md) | Package layout, data I/O, two-stage MAP, holdout open-loop metric, Laplace UQ, CSV examples |

## What the identifier is

A grey-box **stochastic differential equation** in JAX. Indoor temperature is the **measurement**, not the score. HVAC **on/off (interval runtime)** is observed; rated capacity $`Q_{\mathrm{rated}}`$ is optional. Default protocol: `--q-rated unknown`.

A Kalman mean that tracks the thermostat series is **not** a success metric. The dynamics check is a chronological **holdout open-loop** rollout that sees weather and runtime (or metered kW), not a filter overlay of the same $T$.

## Units

Time is in hours so that $`C\,\mathrm{d}T/\mathrm{d}t`$ has units of power:

| Quantity | Unit |
| --- | --- |
| $C$ | $`\mathrm{kWh\,K}^{-1}`$ |
| $R$ | $`\mathrm{K\,kW}^{-1}`$ ($`UA = 1/R`$) |
| Heat fluxes | $`\mathrm{kW}`$ |
| $`Q_{\mathrm{rated}}`$ | $`\mathrm{kW}`$ |
| $`u_{\mathrm{on}}`$ | runtime fraction in $`[0,1]`$ |

## Code map

| Module | Role |
| --- | --- |
| `pinn_building.physics` | Deterministic 1R1C-G energy balance |
| `pinn_building.sde` | SDE, interval-average Kalman, open-loop rollout |
| `pinn_building.model` | Softplus parameters, neural remainder, penalties |
| `pinn_building.train` | Two-stage MAP, chronological split |
| `pinn_building.uq` | Train-only Laplace Hessian |
| `pinn_building.io` | CSV aliases and checkpoints |
| `pinn_building.synthetic` | Digital-twin plant (evaluation only for true $C$, $R$) |

Install and run: see the repository [README](../README.md). CSV examples: [examples/README.md](../examples/README.md).
