# Identifiability of lumped building thermal parameters from smart-thermostat runtime

This package fits a gray-box **stochastic differential equation** in JAX to weather and smart-thermostat time series. The question is whether lumped capacitance `C` [kWh/K], envelope resistance `R` [K/kW], and rated HVAC capacity `Q_rated` [kW] can be recovered when the HVAC channel is **interval runtime**, not delivered HVAC power.

Commercial thermostats report a signed duty cycle and an interval-mean indoor temperature. They do not report kilowatts, and they do not sample `T` at an instant. Indoor temperature is the **measurement**, not the score: a filter that sees `T` can track `T` even when `C` and `R` are wrong. The checks used here are recovery of the physical parameters against digital-twin truth and chronological **holdout open-loop** `T`.

By default (`--q-rated unknown`) the identifier sees `u = hvac_on_frac ∈ [-1, 1]` and learns a positive rated capacity `Q_rated` (softplus, the same parameterization as `C` and `R`). The HVAC term in the drift is then

```
Q_hvac = Q_rated * u
```

Heating-only files with `u ∈ [0, 1]` work as before. Cooling-only files can use a `cooling_on` column (already stored as negative `u`) or `--hvac-mode cooling` on unsigned generic on/off. Reverse-cycle files with both `heating_on` and `cooling_on` become `u = u_heat - u_cool`. Rated cooling and heating still share one `Q_rated` magnitude.

The older `--q-rated known` protocol is **optimistic**: the identifier is given plant `q_hvac_kw` (positive when heating, negative when cooling). Real ecobee-style exports do not provide that. If cooling power is stored as a positive magnitude next to a `cooling_on` column, it is aligned to the sign of runtime.

HVAC on/off is **observed**; it is not inferred as a switching process. Rated capacity is what may be unknown. The same SDE is used for heating and cooling; only the sign of `u` changes.

## Identifiability

On heating intervals, `Q_rated` and `C` both scale `Q/C` in `dT/dt = (Q_rated u)/C + …`, so they alias if the equipment is rarely off. The same alias appears on cooling intervals with `u ≤ 0`. Occupied and setback **off** intervals give a free-response time constant (`RC` / `UA/C`). The map `C, Q_rated → α·`, `R → R/α` also leaves the wind-inflated loss `UA_eff/C = (1+k v)/(RC)` invariant; solar, occupancy, and the scalar moisture term `β(ω_a−ω_i)/C` break an exact symmetry and leave an ill-conditioned valley. On the default seven-day winter twin, unknown-`Q_rated` MAP errors are large (`C` about 43%, `R` about 67%, `Q_rated` about 42%) while holdout open-loop `T` RMSE stays around 0.13 K — lower than the metered-`Q_hvac` holdout RMSE of about 0.22 K. That is why indoor `T` is not the score. Metered HVAC power recovers winter `C` and `R` but does **not** recover winter solar aperture (~46%) or summer `R` (~19%) when solar aperture is co-estimated. A same-budget remainder-off gray-box fit leaves the same qualitative limits (runtime `Q_rated`–`C` aliasing; summer `R` bias). The remainder penalty is Pearson `r²` (linear leakage only). The remainder is a bounded micro-residual (`|r|≤0.18 kW`). Humidity ratios use the observed interval `T` (not 1-minute plant `T`); `β` is frozen at the prior. Training stays two-stage, the holdout is chronological, hidden `Q_int` is unused, and the twin is **not** given easier excitation to make the alias disappear. These are observability limits under one week of closed-loop hysteresis, not a claim about multi-week or actively excited datasets.

## Evaluation protocol (no leakage)

The default example is one contiguous digital-twin week (same generator, same seed and config). The **last 2 of 7 days** are holdout; the first 5 days are train. Shorter series fall back to the last 30%. The plant still uses 9 kW internally to generate data; that value does **not** enter unknown-mode training, remainder features, Kalman inputs, or holdout open-loop except as signed runtime `u`.

| Rule | What we do |
| --- | --- |
| Split | Chronological prefix / suffix only. Never shuffle timesteps, never random row splits. |
| Fit | Train Kalman NLL only. Holdout is **not** used to fit `C`, `R`, `Q_rated`, remainder, Fourier `μ_q`, or noise. No peeking at holdout loss for hyperparameters. Optional val would have to be strictly before holdout; default is train + holdout only. |
| Features | Remainder and Fourier occupancy see only information available at interval `k`: weather, **HVAC on/off** (or metered HVAC power if `--q-rated known`), clock. No future weather/`T`/HVAC. No indoor `T` as a remainder feature. No delivered HVAC power in unknown mode. |
| Hidden occupancy | `q_int_kw_hidden` is written on the synthetic CSV for plots. It is **never** a training feature, Kalman input, remainder input, or loss term. `Q_int` stays a latent OU. True `BuildingParams` are evaluation-only. |
| Primary `T` metric | **Holdout open-loop** (physics + train-fit remainder/`μ_q`): start from the last **train** filter state, roll out interval-average `T` with holdout **weather + estimated `Q_rated` × holdout on/off** (unknown mode) or known kW (optimistic mode). This is a dynamics test, not “filter tracks `T`”. |
| Parameter metric | MAP `C`, `R`, and (unknown mode) `Q_rated` vs synthetic truth (eval only) and Laplace intervals from the **train** likelihood only. |
| Secondary | Train mean Kalman NLL. Do not headline in-sample Kalman `T` RMSE. |

## Two-stage identification

The neural remainder can absorb UA and `1/C` if it is free while `C`, `R`, and `Q_rated` are still moving. Training on the train prefix is therefore staged:

1. **Stage A** — remainder **off** (frozen at 0, last layer zero-init). Fit `{C, R, Q_rated (unknown mode), A_s, β, σ, κ, Fourier μ_q}` on train. `Q_rated` is initialized at 6 kW, not plant truth 9 kW.
2. **Stage B1** — unfreeze remainder with a **stronger** identifiability penalty and smaller LR; **freeze `C`, `R`, and `Q_rated`**.
3. **Stage B2** — jointly fine-tune with the same remainder penalty.

Fourier `μ_q` and `σ_q` have an L2 / log prior so latent occupancy cannot freely alias into UA and `1/C`. Occupancy is not clamped to the hidden series.

## Energy-balance SDE

Indoor temperature $T$ and a latent internal-gain / occupancy state $Q_{\mathrm{int}}$:

$$
\begin{aligned}
\mathrm{d}T
&=
\frac{1}{C}\bigl[
UA_{\mathrm{eff}}(T_a-T)+A_s I+\beta(\omega_a-\omega_i)
+Q_{\mathrm{rated}} u+Q_{\mathrm{int}}+r_\theta(\boldsymbol{\phi})
\bigr]\,\mathrm{d}t
+\sigma_T(\boldsymbol{\phi})\,\mathrm{d}W_T, \\
\mathrm{d}Q_{\mathrm{int}}
&=
\kappa\bigl(\mu(t)-Q_{\mathrm{int}}\bigr)\,\mathrm{d}t
+\sigma_q\,\mathrm{d}W_Q.
\end{aligned}
$$

| Symbol | Role |
| --- | --- |
| `C`, `R` | Learnable physical parameters (`UA_eff = (1/R)(1 + k v_wind)`) |
| `u` | Observed signed HVAC runtime in `[-1, 1]` (heat +, cool −) |
| `Q_rated` | Constant rated HVAC capacity [kW]; learned in unknown mode, not estimated when HVAC power is metered. Same magnitude for heat and cool. |
| `Q_hvac` | Delivered HVAC power into the node [kW]: `Q_rated u` (unknown) or metered plant kW (known, optimistic; negative when cooling) |
| `Q_int` | Unmeasured occupancy/appliance gain (OU process) |
| `r_θ(φ)` | Small neural remainder of exogenous weather/HVAC-on, **identifiability-constrained** (Pearson `r²`; linear leakage only) so it cannot absorb UA or `1/C` linearly |
| `σ_T(φ)` | Input-dependent neural diffusion (state-independent, so Kalman stays Gaussian) |
| `μ(t)` | Daily Fourier mean occupancy |

**Observation model (thermostat interval average):**

$$
\bar y_k
=
\frac{1}{\Delta}\int_{t_{k-1}}^{t_k} T(s)\,\mathrm{d}s
+ e_k,
\qquad
e_k\sim\mathcal{N}(0,\sigma_y^2).
$$

This is not a point sample $T(t_k)$. The **training** likelihood is the Gaussian Kalman filter of the Euler–Maruyama discretization, observing the **post-step** mean of the 1-minute substeps in each 5-minute interval. The digital-twin exporter averages incoming (pre-update) 1-minute states; the one-substep offset is far below sensor noise. The **holdout `T` metric** does not use that update.

## Uncertainty quantification

Laplace / observed information is computed on **train only**:

1. Objective = **sum** of train interval NLLs + scaled priors / identifiability penalty (same critical points as the mean-NLL trainer; Hessian is not `mean NLL / N`).
2. Joint unconstrained Hessian over `{C, R, Q_rated (unknown mode), A_s, β, σ_T, σ_q, σ_y, κ}` **and** Fourier `μ_q` coefficients. Neural remainder/diffusion **weights stay at MAP** (not included).
3. Delta method back to positive units; 95% local intervals for the physical parameters when the Hessian is positive definite. These are not calibrated frequentist CIs. `C` and `R` intervals are from this joint train Hessian — not a silent 2-parameter slice mixed with a mean-NLL joint Hessian.

Intervals **condition on neural weights at MAP** (and include occupancy priors), so they can be narrower than a full profile over the nets. They are still the correct *scale* (sum of train NLL, train series only). Filter bands on train `T`/`Q_int` are state uncertainty, not holdout accuracy.

## Literature positioning

### Already standard

- Continuous-time stochastic RC models with Wiener process noise, Kalman/EKF maximum likelihood, and CTSM-R: [Madsen & Holst 1995](https://www.sciencedirect.com/science/article/abs/pii/037877889400909M), [Bacher & Madsen 2011](https://www.sciencedirect.com/science/article/abs/pii/S0378778811000470), [Kristensen, Madsen & Jørgensen 2004](https://www.sciencedirect.com/science/article/abs/pii/S0005109803003123), [CTSM-R building example](https://ctsm.info/building2.pdf). Measurement equations are almost always `Y_k = T_i(t_k) + e_k` (point samples). HVAC/heat input `Φ_h` is an exogenous input when **metered**.
- Smart-thermostat gray-box / data-driven models using **runtime as a known feature** (not a latent mode): e.g. [Huchuk, Sanner & O’Brien](https://ssanner.github.io/papers/jbps21_resthermal.pdf); ecobee 5-minute telemetry ([runtime report](https://www.ecobee.com/home/developer/api/documentation/v1/operations/get-runtime-report.shtml) — note the API documents interval *start* values, not always means; exports still bundle runtime over the interval). Runtime is not HVAC power unless rated capacity is assumed or learned.
- Inverse PINNs for building RC parameters, including thermostat data: [Gokhale et al. 2022](https://www.sciencedirect.com/science/article/pii/S0306261922003946), [Kim et al. 2025](https://publications.ibpsa.org/proceedings/bs/2025/papers/bs2025_1471.pdf). Deterministic energy-balance residuals, not SDEs.
- Hybrid gray-box + neural remainder in JAX: [IBPSA BS2023 1641](https://publications.ibpsa.org/proceedings/bs/2023/papers/bs2023_1641.pdf). Parameter UQ in classical CTSM is MLE ± sd from the information matrix; fully Bayesian RC (Stan/PyMC) exists but is heavier. Neural-SDE Bayesian UQ is not settled.

### What this package implements

The **combination** that we could not find as a published, reusable stack:

1. Physics RC **SDE drift** with HVAC as **observed signed runtime** `Q_rated u` (capacity optional / learned),
2. **Latent OU occupancy** rather than white noise only on `T`,
3. **Post-step interval-mean** observation operator on 1-minute Euler–Maruyama maps (not a Dirac sample of `T` at `t_k`),
4. Two-stage, identifiability-constrained neural remainder + input-dependent `σ_T(u)`,
5. JAX autodiff through the Kalman path likelihood, plus **train-only Laplace UQ** on `{C,R,Q_rated,…}` using the **sum** NLL Hessian,
6. Chronological holdout **open-loop `T`** as the dynamics check (not in-sample filter tracking).

That is a **methods demo**, not a claim that interval averaging or OU gains have never been mentioned separately. The closest published pieces are CTSM point-sample SDEs, PINN-RC without process noise, and hybrid NN correctors without a Gaussian SDE likelihood.

HVAC hysteresis is **not** treated as latent (closed-loop identifiability of switching SDEs is a different, already-discussed problem). On/off is observed; capacity may not be.

## Documentation

Mathematical models and the identification framework:

- Repository: [`docs/`](docs/README.md)
- Wiki: [github.com/smyng91/pi-nsde-building-thermal/wiki](https://github.com/smyng91/pi-nsde-building-thermal/wiki)

## Install and run

Python 3.10+ and CPU JAX. Install [uv](https://docs.astral.sh/uv/), then from the repo root:

```bash
git clone https://github.com/smyng91/pi-nsde-building-thermal.git
cd pi-nsde-building-thermal
uv sync --extra dev
uv run pytest -q
```

`uv run` uses the project environment on Windows, macOS, and Linux (no venv activation, no OS-specific Python paths).

Package demo (digital twin, unknown $Q_{\mathrm{rated}}$):

```bash
uv run python -m pi_nsde_building_thermal.example --q-rated unknown
```

Custom CSV (see [examples/README.md](examples/README.md)):

```bash
uv run python examples/generate_synthetic.py
uv run python examples/train_csv.py output/synthetic_thermostat.csv --q-rated unknown
uv run python examples/infer_csv.py output/synthetic_thermostat.csv output/checkpoint.pkl --mode holdout
```

`--q-rated unknown` is the default. `--q-rated known` reproduces the optimistic metered-`Q_hvac` case. `--q-rated both` runs unknown then known into the same JSON (`q_rated_runs`). Add `--hvac-mode cooling` on the demo for the summer twin.

Outputs in `outputs/`: `synthetic_timeseries.csv` (includes a `split` column; `q_int_kw_hidden` is eval-only; `hvac_kw` is plant truth / eval), `parameter_estimates.json` (MAP, train Laplace sd/CI, holdout open-loop RMSE/MAE, `known_kw_reference` when running unknown), `identification.png`, `training_history.csv` (stage column).

Python API (after `uv sync --extra dev`):

```python
from pi_nsde_building_thermal.synthetic import generate_synthetic_building
from pi_nsde_building_thermal.train import TrainConfig, identify_building
from pi_nsde_building_thermal.uq import quantify_uncertainty

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
