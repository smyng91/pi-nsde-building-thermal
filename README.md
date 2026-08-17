# Physics-informed neural SDE for building C and R

Grey-box **stochastic differential equation** in JAX that estimates a building's effective thermal capacity `C` [kWh/K] and envelope resistance `R` [K/kW] from ambient weather and smart-thermostat timeseries.

A thermostat exposes **on/off / interval runtime**, not delivered HVAC kilowatts. Default identification (`--q-rated unknown`) sees `u_on = hvac_on_frac ∈ [0, 1]` only and learns a positive rated capacity `Q_rated` (softplus, same style as `C`, `R`). Drift HVAC term:

```
Q_hvac = Q_rated * u_on
```

`--q-rated known` is the older **optimistic** protocol: the identifier is given plant `q_hvac_kw` (true 9 kW × runtime). Real ecobee-style exports do not provide that.

HVAC on/off is **observed**. It is not inferred as a switching process. Capacity is what may be unknown.

Indoor temperature is the **measurement**, not the result. Overlaying a Kalman mean on the thermostat series is **not** a success metric: a filter that sees `T` can track `T` even when `C` and `R` are wrong.

## Identifiability (honest)

On heating intervals, `Q_rated` and `C` both scale `Q/C` in `dT/dt = (Q_rated u_on)/C + …`, so they alias if the heater is rarely off. Occupied/setback **off** intervals give a free-response time constant (`RC` / `UA/C`). On the default 7-day winter twin, unknown-`Q_rated` MAP errors are large (`C` ~47%, `Q_rated` ~45%) while holdout open-loop `T` RMSE stays ~0.11 K — that is why indoor `T` is not the score. Wind-inflated UA and occupancy structure give extra scale, but they do not make `Q_rated` as clean as metered kW. We keep two-stage training, chronological holdout, no hidden `Q_int`, and we do **not** fabricate easier excitation.

## Evaluation protocol (no leakage)

Default example: one contiguous digital-twin week (same generator, same seed/config). **Last 2 of 7 days** are holdout; the first 5 days are train. Shorter series fall back to the last 30%. The plant still uses 9 kW internally to generate data; that value does **not** enter unknown-mode training, remainder features, Kalman inputs, or holdout open-loop except as `u_on`.

| Rule | What we do |
| --- | --- |
| Split | Chronological prefix / suffix only. Never shuffle timesteps, never random row splits. |
| Fit | Train Kalman NLL only. Holdout is **not** used to fit `C`, `R`, `Q_rated`, remainder, Fourier `μ_q`, or noise. No peeking at holdout loss for hyperparameters. Optional val would have to be strictly before holdout; default is train + holdout only. |
| Features | Remainder and Fourier occupancy see only information available at interval `k`: weather, **HVAC on/off** (or metered kW if `--q-rated known`), clock. No future weather/`T`/HVAC. No indoor `T` as a remainder feature. No delivered kW in unknown mode. |
| Hidden occupancy | `q_int_kw_hidden` is written on the synthetic CSV for plots. It is **never** a training feature, Kalman input, remainder input, or loss term. `Q_int` stays a latent OU. True `BuildingParams` are evaluation-only. |
| Primary `T` metric | **Holdout open-loop** (physics + train-fit remainder/`μ_q`): start from the last **train** filter state, roll out interval-average `T` with holdout **weather + estimated `Q_rated` × holdout on/off** (unknown mode) or known kW (optimistic mode). This is a dynamics test, not “filter tracks `T`”. |
| Parameter metric | MAP `C`, `R`, and (unknown mode) `Q_rated` vs synthetic truth (eval only) and Laplace intervals from the **train** likelihood only. |
| Secondary | Train mean Kalman NLL. Do not headline in-sample Kalman `T` RMSE. |

## Two-stage identification

The neural remainder can absorb UA and `1/C` if it is free while `C`, `R`, and `Q_rated` are still moving.

1. **Stage A** — remainder **off** (frozen at 0, last layer zero-init). Fit `{C, R, Q_rated (unknown mode), A_s, β, σ, κ, Fourier μ_q}` on train. `Q_rated` is initialized at 6 kW, not plant truth 9 kW.
2. **Stage B1** — unfreeze remainder with a **stronger** identifiability penalty and smaller LR; **freeze `C`, `R`, and `Q_rated`**.
3. **Stage B2** — jointly fine-tune with the same remainder penalty.

Fourier `μ_q` and `σ_q` have an L2 / log prior so latent occupancy cannot freely alias into UA and `1/C`. Occupancy is not clamped to the hidden series.

## Energy-balance SDE

Indoor temperature `T` and a latent internal-gain / occupancy state `Q_int`:

```
dT      = (1/C) [ UA_eff (T_a - T) + A_s I + β (ω_a - ω_i) + Q_rated u_on + Q_int + r_θ(u) ] dt
          + σ_T(u) dW_T
dQ_int  = κ (μ(t) - Q_int) dt + σ_q dW_q
```

| Symbol | Role |
| --- | --- |
| `C`, `R` | Learnable physical parameters (`UA_eff = (1/R)(1 + k v_wind)`) |
| `u_on` | Observed interval HVAC runtime fraction in `[0, 1]` |
| `Q_rated` | Learnable rated capacity [kW] in unknown mode; skipped when kW is given |
| `Q_hvac` | `Q_rated u_on` (unknown) or metered kW (known, optimistic) |
| `Q_int` | Unmeasured occupancy/appliance gain (OU process) |
| `r_θ(u)` | Small neural remainder of exogenous weather/HVAC-on, **identifiability-constrained** so it cannot absorb UA or `1/C` |
| `σ_T(u)` | Input-dependent neural diffusion (state-independent, so Kalman stays Gaussian) |
| `μ(t)` | Daily Fourier mean occupancy |

**Observation model (thermostat interval average):**

```
ȳ_k = (1/Δ) ∫_{t_{k-1}}^{t_k} T(s) ds + e_k ,    e_k ~ N(0, σ_y²)
```

not a point sample `T(t_k)`. The **training** likelihood is the Gaussian Kalman filter of the Euler–Maruyama discretisation, observing the mean of the substeps in each 5-minute interval. The **holdout `T` metric** does not use that update.

## Uncertainty quantification

Laplace / observed information on **train only**:

1. Objective = **sum** of train interval NLLs + scaled priors / identifiability penalty (same critical points as the mean-NLL trainer; Hessian is not `mean NLL / N`).
2. Joint unconstrained Hessian over `{C, R, Q_rated (unknown mode), A_s, β, σ_T, σ_q, σ_y, κ}` **and** Fourier `μ_q` coefficients. Neural remainder/diffusion **weights stay at MAP** (not included).
3. Delta method back to positive units; 95% CIs for the physical parameters. `C` and `R` intervals are from this joint train Hessian — not a silent 2-parameter slice mixed with a mean-NLL joint Hessian.

Intervals **condition on neural weights at MAP** (and include occupancy priors), so they can be narrower than a full profile over the nets. They are still the correct *scale* (sum of train NLL, train series only). Filter bands on train `T`/`Q_int` are state uncertainty, not holdout accuracy.

## Literature positioning

### Already standard

- Continuous-time stochastic RC models with Wiener process noise, Kalman/EKF maximum likelihood, and CTSM-R: [Madsen & Holst 1995](https://www.sciencedirect.com/science/article/abs/pii/037877889400909M), [Bacher & Madsen 2011](https://www.sciencedirect.com/science/article/abs/pii/S0378778811000470), [Kristensen, Madsen & Jørgensen 2004](https://www.sciencedirect.com/science/article/abs/pii/S0005109803003123), [CTSM-R building example](https://ctsm.info/building2.pdf). Measurement equations are almost always `Y_k = T_i(t_k) + e_k` (point samples). HVAC/heat input `Φ_h` is an exogenous input when **metered**.
- Smart-thermostat grey-box / data-driven models using **runtime as a known feature** (not a latent mode): e.g. [Huchuk, Sanner & O’Brien](https://ssanner.github.io/papers/jbps21_resthermal.pdf); ecobee 5-minute telemetry ([runtime report](https://www.ecobee.com/home/developer/api/documentation/v1/operations/get-runtime-report.shtml) — note the API documents interval *start* values, not always means; exports still bundle runtime over the interval). Runtime is not delivered kW unless capacity is assumed or learned.
- Inverse PINNs for building RC parameters, including thermostat data: [Gokhale et al. 2022](https://www.sciencedirect.com/science/article/pii/S0306261922003946), [Kim et al. 2025](https://publications.ibpsa.org/proceedings/bs/2025/papers/bs2025_1471.pdf). Deterministic energy-balance residuals, not SDEs.
- Hybrid grey-box + neural remainder in JAX: [IBPSA BS2023 1641](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1641.pdf). Parameter UQ in classical CTSM is MLE ± sd from the information matrix; fully Bayesian RC (Stan/PyMC) exists but is heavier. Neural-SDE Bayesian UQ is not settled.

### What we implement (honest contribution, not a claim of a new theorem)

The **combination** that we could not find as a published, reusable stack:

1. Physics RC **SDE drift** with HVAC as **observed on/off** times `Q_rated u_on` (capacity optional / learned),
2. **Latent OU occupancy** rather than white noise only on `T`,
3. **Interval-average** observation operator (integrated sampling of `T` over the thermostat interval),
4. Two-stage, identifiability-constrained neural remainder + input-dependent `σ_T(u)`,
5. JAX autodiff through the Kalman path likelihood, plus **train-only Laplace UQ** on `{C,R,Q_rated,…}` using the **sum** NLL Hessian,
6. Chronological holdout **open-loop `T`** as the dynamics check (not in-sample filter tracking).

That is a **methods demo**, not a claim that interval averaging or OU gains have never been mentioned separately. Closest pieces: CTSM point-sample SDEs; PINN-RC without process noise; hybrid NN correctors without a Gaussian SDE likelihood.

We deliberately **do not** treat HVAC hysteresis as latent (closed-loop identifiability of switching SDEs is a different, already-discussed problem). On/off is observed; capacity may not be.

## Install and run

Python 3.10+, CPU JAX. Clone and install:

```bash
git clone git@github.com:smyng91/pi-nde-building-thermal.git
cd pi-nde-building-thermal
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Package demo (digital twin, unknown \(Q_\mathrm{rated}\)):

```bash
.\.venv\Scripts\python.exe -m pinn_building.example --q-rated unknown
```

Custom CSV (see [examples/README.md](examples/README.md)):

```bash
.\.venv\Scripts\python.exe examples/generate_synthetic.py
.\.venv\Scripts\python.exe examples/train_csv.py output/synthetic_thermostat.csv --q-rated unknown
.\.venv\Scripts\python.exe examples/infer_csv.py output/synthetic_thermostat.csv output/checkpoint.pkl --mode holdout
```

`--q-rated unknown` is the default. `--q-rated known` reproduces the optimistic metered-kW case. `--q-rated both` runs unknown then known into the same JSON (`q_rated_runs`).

Outputs in `outputs/`: `synthetic_timeseries.csv` (includes a `split` column; `q_int_kw_hidden` is eval-only; `hvac_kw` is plant truth / eval), `parameter_estimates.json` (MAP, train Laplace sd/CI, holdout open-loop RMSE/MAE, `known_kw_reference` when running unknown), `identification.png`, `training_history.csv` (stage column).

```python
from pinn_building.synthetic import generate_synthetic_building
from pinn_building.train import TrainConfig, identify_building
from pinn_building.uq import quantify_uncertainty

data = generate_synthetic_building()
ident = identify_building(data.arrays, TrainConfig(q_rated="unknown"), holdout_days=2.0)
uq = quantify_uncertainty(
    ident.fit.params,
    ident.train,
    ident.fit.filter,
    TrainConfig(q_rated="unknown"),
    ident.fit.n_sub,
    remainder_gate=ident.fit.remainder_gate,
    lambda_id=ident.fit.lambda_id,
)
print(ident.fit.estimated, ident.holdout_rmse, uq.laplace.sd[:3])
```
