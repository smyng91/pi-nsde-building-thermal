# Mathematical models

This page specifies the gray-box RC model used to identify an occupied house: a lumped energy-balance **drift**, a latent occupancy **SDE**, and a thermostat **interval-average** observation. HVAC **signed runtime** is an observed input. Rated capacity may be known or learned. Implementation: `pi_nsde_building_thermal.physics`, `pi_nsde_building_thermal.sde`.

How these equations are fitted is in [framework](framework.md).

## 1. Lumped energy balance

The plant is a single indoor node with effective capacity $C$ and envelope resistance $R$:

$$
C\,\frac{\mathrm{d}T}{\mathrm{d}t}
=
UA_{\mathrm{eff}}(T_a-T)
+ A_s I
+ \beta(\omega_a-\omega_i)
+ Q_{\mathrm{hvac}}
+ Q_{\mathrm{int}}.
$$

| Symbol | Meaning | Code |
| --- | --- | --- |
| $T$ | Indoor air temperature | `t_in_c` |
| $T_a$ | Outdoor dry-bulb | `t_out_c` |
| $I$ | Global horizontal irradiance | `ghi_w_m2 / 1000` $\rightarrow$ $\mathrm{kW\,m}^{-2}$ in $A_s I$ |
| $\omega_a,\omega_i$ | Outdoor / indoor humidity ratio | `humidity_ratio` |
| $Q_{\mathrm{hvac}}$ | HVAC heat into the node | see §4 |
| $Q_{\mathrm{int}}$ | Unmeasured internal gains | latent state |
| $A_s$ | Effective solar aperture $\mathrm{m}^2$ | `A_s` |
| $\beta$ | Moisture-driven heat coefficient $\mathrm{kW}$ per $\mathrm{kg\,kg}^{-1}$ | `beta` |

Envelope conductance is wind-inflated with a **fixed** coefficient $k_v = 0.04\,\mathrm{(m/s)}^{-1}$ (not a free parameter, so wind does not alias $R$):

$$
UA_{\mathrm{eff}}
=
\frac{1}{R}\bigl(1 + k_v v\bigr).
$$

Humidity ratio uses Magnus–Tetens saturation vapor pressure at atmospheric pressure $101325\,\mathrm{Pa}$. Indoor RH on the digital twin is held at a constant (default $0.40$); outdoor RH is an exogenous series. The coefficient $\beta$ is a **scalar** gray-box lump ($\mathrm{kW}$ per $\mathrm{kg\,kg}^{-1}$), not $\dot{m}_{\mathrm{inf}}(t)\,h_{fg}$. It is not indexed by wind or by $UA_{\mathrm{eff}}$, so latent load is decoupled from wind-driven infiltration. The identifier **freezes** $\beta$ at plant truth (default $120$, matching the digital-twin coefficient); it is not a MAP parameter.

The result is a **1R1C-G** node (one resistance, one capacitance, plus gains). There is no second envelope node (2R2C), no ducts, no multi-zone air, and no latent HVAC coil.

## 2. Physics-informed SDE

Process noise is written on both temperature and occupancy. With Wiener processes $W_T$ and $W_Q$:

$$
\begin{aligned}
\mathrm{d}T
&=
\frac{1}{C}\Bigl[
UA_{\mathrm{eff}}(T_a-T)+A_s I+\beta(\omega_a-\omega_i)
+Q_{\mathrm{hvac}}+Q_{\mathrm{int}}+r_\theta(\boldsymbol{\phi})
\Bigr]\mathrm{d}t
+\sigma_T(\boldsymbol{\phi})\,\mathrm{d}W_T, \\
\mathrm{d}Q_{\mathrm{int}}
&=
\kappa\bigl(\mu(t)-Q_{\mathrm{int}}\bigr)\,\mathrm{d}t
+\sigma_q\,\mathrm{d}W_Q.
\end{aligned}
$$

$Q_{\mathrm{int}}$ is an Ornstein–Uhlenbeck process that mean-reverts to a daily Fourier mean $\mu(t)$. It is **latent**: the identifier never reads a hidden occupancy column.

The neural remainder $r_\theta$ and the diffusion scale $\sigma_T(\boldsymbol{\phi})$ see only **contemporaneous exogenous** features (weather, HVAC channel, clock). They do **not** see $T$ or $Q_{\mathrm{int}}$, so the filter stays a linear-Gaussian SDE in the state.

### Occupancy mean

With unconstrained Fourier coefficients $\mathbf{f}$ (length 7: DC + three harmonics),

$$
\mu(t)
=
\mathrm{softplus}\!\left(
f_0
+ \sum_{k=1}^{3}
\bigl(f_{2k-1}\sin(2\pi k t/24)+f_{2k}\cos(2\pi k t/24)\bigr)
\right).
$$

Units: $\mu$ and $Q_{\mathrm{int}}$ in $\mathrm{kW}$; $\kappa$ in $\mathrm{h}^{-1}$; $\sigma_T$ in $\mathrm{K\,h}^{-1/2}$; $\sigma_q$ in $\mathrm{kW\,h}^{-1/2}$.

### Remainder and diffusion nets

Let $\boldsymbol{\phi}_k$ be the contemporaneous exogenous features and $\mathbf{z}_k$ their $z$-score on the **train** prefix only:

$$
\begin{aligned}
\boldsymbol{\phi}_k
&=
\bigl(
T_a,\; I,\; \mathrm{RH}_a,\; v,\;
\mathrm{HVAC}_k,\;
\sin\omega_k,\;\cos\omega_k,\;
\sin 2\omega_k,\;\cos 2\omega_k
\bigr), \\
\omega_k&=2\pi t_k/24, \\
\mathbf{z}_k
&=
(\boldsymbol{\phi}_k-\bar{\boldsymbol{\phi}}_{\mathrm{train}})/\boldsymbol{\sigma}_{\mathrm{train}}.
\end{aligned}
$$

$\mathrm{HVAC}_k = u_{k}$ (signed runtime) in unknown-$Q_{\mathrm{rated}}$ mode, or metered $Q_{\mathrm{hvac}}$ in known mode.

$$
\begin{aligned}
r_\theta(\boldsymbol{\phi})
&=
g\cdot 0.18\tanh(\mathrm{MLP}_{16,16}(\mathbf{z})), \\
\sigma_T(\boldsymbol{\phi})
&=
\sigma_T^{\mathrm{base}}
\exp\bigl(\tfrac12 g\cdot 0.35\,\mathrm{clip}(\mathrm{MLP}_{8,8}(\mathbf{z}),-2,2)\bigr).
\end{aligned}
$$

Gate $g=0$ in Stage A (remainder off, last layer zero-initialized). Indoor $T$ is not a remainder feature. The $0.18$ prefactor caps $\lvert r_\theta\rvert$ at $0.18\,\mathrm{kW}$; plant occupancy peaks near $1.5\,\mathrm{kW}$ and HVAC is $9\,\mathrm{kW}$, so the remainder is a bounded residual, not a second occupancy model. The twin plant uses constant $\sigma_T$. Train-prefix RMS / $\sigma_T(\phi)$ summaries are in `remainder_diagnostics`.

## 3. Interval-average observation

A thermostat export is treated as an **interval statistic** over $\Delta$ (default $5\,\mathrm{min}$), not as a point sample $T(t_k)$. The continuous ideal is

$$
\bar y_k
=
\frac{1}{\Delta}\int_{t_{k-1}}^{t_k} T(s)\,\mathrm{d}s
+ e_k,
\qquad
e_k\sim\mathcal{N}(0,\sigma_y^2).
$$

The digital twin integrates Euler--Maruyama at $1\,\mathrm{min}$ and exports the arithmetic mean of the five incoming (pre-update) 1-minute states on each reporting interval. The identifier uses the same 1-minute EM maps but observes the mean of the $n_{\mathrm{sub}}$ post-step temperatures---a one-substep index offset of order $0.02\,\mathrm{K}$, below the $0.07\,\mathrm{K}$ sensor noise. Intra-interval HVAC switches are collapsed to the exported 5-minute mean runtime (ZOH). An exact continuous-discrete interval mean (matrix exponential / Van Loan) is not used, because the twin itself is 1-minute Euler--Maruyama. Indoor humidity ratio $\omega_i$ is recomputed from the **observed** interval $T$ and RH after sensor noise (thermostat analogue), not from 1-minute plant $T$. The coefficient $\beta$ is frozen at plant truth ($120$); it is not a recovery target.

Weather and HVAC are held ZOH over the interval (as in a typical 5-minute export).

## 4. HVAC: observed on/off, optional capacity

HVAC hysteresis is **not** modeled as a latent switching SDE. Runtime is an observed signed fraction $u\in[-1,1]$: positive is heating, negative is cooling. Rated capacity $Q_{\mathrm{rated}}$ stays positive. Unsigned heating datasets use $u\in[0,1]$ as before. Unsigned cooling runtime (`--hvac-mode cooling`, or a `cooling_on` column) is stored as $u\le 0$. Reverse-cycle files with both heat and cool columns become $u = u_{\mathrm{heat}}-u_{\mathrm{cool}}$.

| Protocol | Drift HVAC term | What the identifier must not use |
| --- | --- | --- |
| **Unknown** $Q_{\mathrm{rated}}$ (default) | $Q_{\mathrm{hvac}}=Q_{\mathrm{rated}}\,u$ | Plant `q_hvac_kw` / true rated kW |
| **Known** kW (optimistic) | interval-mean delivered $Q_{\mathrm{hvac}}$ (negative while cooling) | — |

Positive parameters use a shifted softplus, e.g. $C=\mathrm{softplus}(c_{\mathrm{raw}})+0.3$. Prior / init for $Q_{\mathrm{rated}}$ is $6\,\mathrm{kW}$, not plant truth $9\,\mathrm{kW}$.

## 5. Euler–Maruyama and the Kalman likelihood

State $\mathbf{x}=[T,Q_{\mathrm{int}}]^{\top}$. Over a substep $\delta=\Delta/n_{\mathrm{sub}}$ (default $n_{\mathrm{sub}}=5$):

$$
\mathbf{x}'
=
F_2\mathbf{x}+\mathbf{g}_2+\boldsymbol{\varepsilon},
\qquad
F_2
=
\begin{pmatrix}
1+\delta a_T & \delta/C \\
0 & 1-\delta\kappa
\end{pmatrix},
\quad
a_T=-UA_{\mathrm{eff}}/C,
$$

$$
\mathbf{g}_2
=
\delta
\begin{pmatrix}
\bigl(UA_{\mathrm{eff}}T_a+A_s I+\beta(\omega_a-\omega_i)+Q_{\mathrm{hvac}}+r_\theta(\boldsymbol{\phi})\bigr)/C \\
\kappa\mu(t)
\end{pmatrix}.
$$

The last row of $F_3$ equals the first row of $F_2$ in the $T,Q$ columns, with a $1$ on $S$, because $S'=S+T'$.

$$
\bar y = S/n_{\mathrm{sub}} + e,
$$

that is, observation vector $\mathbf{h}=[0,0,1/n_{\mathrm{sub}}]$ on the augmented state $[T,Q_{\mathrm{int}},S]$. The training objective is the Gaussian innovation NLL of this Kalman filter on the **train prefix only**.

Open-loop scoring uses the same mean EM maps with **no Kalman update** and $\sigma_T=\sigma_q=0$ in the rollout (a deterministic mean trajectory), starting from the last train filter state.

## 6. Identifiability

On intervals where HVAC is on (heating or cooling),

$$
\frac{\mathrm{d}T}{\mathrm{d}t}
\supset
\frac{Q_{\mathrm{rated}}\,u}{C}.
$$

$C$ and $Q_{\mathrm{rated}}$ **scale together** on heating (or cooling) intervals. Off and setback intervals supply a free-response time constant $\sim RC$. The map $C\mapsto\alpha C$, $Q_{\mathrm{rated}}\mapsto\alpha Q_{\mathrm{rated}}$, $R\mapsto R/\alpha$ leaves $\tau=RC$, $\eta=Q_{\mathrm{rated}}/C$, and the wind-inflated loss $UA_{\mathrm{eff}}/C=(1+k_v v)/(RC)$ invariant. It is an exact symmetry of the homogeneous envelope-plus-HVAC drift, not of the fitted SDE: $A_s I/C$, $Q_{\mathrm{int}}/C$, and $\beta(\omega_a-\omega_i)/C$ scale as $1/\alpha$ unless $A_s$ is co-scaled with $C$. Under typical thermostat SNR the likelihood is an ill-conditioned valley along $\alpha$ under first-order Adam, not a Lie-group orbit of equivalent parameters. Cooling uses the same magnitude with $u\le 0$, so $Q_{\mathrm{hvac}}$ extracts heat. Metered HVAC power decouples $C$ from $Q_{\mathrm{rated}}$ but does not by itself identify summer $R$ when $A_s$ is co-estimated.

The neural remainder is penalized so it cannot mimic $UA$, solar aperture, or $Q/C$ **linearly**:

$$
\mathcal{P}_{\mathrm{id}}(r)
=
\mathbb{E}[r^2]
+ \widehat{\mathrm{corr}}^2(r,T_a-T)
+ \widehat{\mathrm{corr}}^2(r,I)
+ \widehat{\mathrm{corr}}^2(r,\mathrm{HVAC}),
$$

where $\widehat{\mathrm{corr}}^2$ is ridge-stabilized Pearson $r^2$ (`pearson_r2` in `model.py`). Pearson does not rule out nonlinear leakage through the MLP; Stage B1 freezes $\{C,R,Q_{\mathrm{rated}}\}$ while the remainder is first unfrozen.

Indoor $T$ enters **only** this train-set regularizer, not remainder features. Fourier $\mu$ and $\sigma_q$ have a weak prior so occupancy cannot freely take up a kW bias.

On the default seven-day winter digital twin with unknown $Q_{\mathrm{rated}}$, MAP $C$ and $Q_{\mathrm{rated}}$ remain aliased (relative errors of tens of percent) while holdout open-loop $T$ RMSE can still be $\sim 0.13\,\mathrm{K}$, lower than the metered-kW holdout RMSE. That is why indoor $T$ is not the identification score. A same-budget remainder-off (gray-box) fit leaves that runtime aliasing intact (absolute $R$ errors need not match).

## 7. Digital-twin plant (evaluation only)

The synthetic generator integrates the **same** energy-balance SDE at $1\,\mathrm{min}$ with a hysteretic thermostat, then exports $5\,\mathrm{min}$ interval means. Default truth (evaluation only; unused in training):

$$
C=9.5\,\mathrm{kWh\,K}^{-1},\quad
R=3.6\,\mathrm{K\,kW}^{-1},\quad
Q_{\mathrm{rated}}=9\,\mathrm{kW},\quad
A_s=8.5\,\mathrm{m}^2.
$$

True $C$, $R$, $Q_{\mathrm{rated}}$ and the hidden $Q_{\mathrm{int}}$ series are **eval-only**. Heating indoor temperature is initialized at $20.0^\circ\mathrm{C}$; cooling at $25.5^\circ\mathrm{C}$.
