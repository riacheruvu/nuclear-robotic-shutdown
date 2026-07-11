# Assumptions & scientific scope

This document is the **truth table** for what `remote_qual` does and does not claim.
If you are new to nuclear engineering, start here — every metric in a report rests on these choices.

> **Bottom line:** this toolkit produces *research qualification evidence* for studying
> remote-robot policies under latency, noise, and a *simplified* radiation field.
> It is **not** a regulatory safety case, **not** a plant ALARA code, and **not** a full
> digital twin of a reactor building.

---

## 1. What problem we model (plain English)

A mobile robot must drive to a **manual valve** in a high-radiation room.
Along the way:

1. **Commands are delayed** (teleoperation / network lag).
2. **Sensors sometimes glitch**.
3. **Wheels slip** a little every step.
4. **Dose accumulates** the longer the robot sits in the field.

We ask two complementary questions:

| Question | Method | Output |
|---|---|---|
| Under bounded noise, could the uncertainty set leave the safe corridor? | Linearized **interval reachability** | `reachability_unsafe` flag |
| How often do missions fail safety (dose or corridor) under random noise? | **Monte Carlo / defensive mixture IS** | `P(fail)`, ESS, CIs |
| How often does the robot finish the job safely? | Nominal-noise rollouts | Mission success rate + 95% CI |

---

## 2. Radiation / dose model

### Formula (research model)

At distance \(r\) (metres) from a point source of activity \(A\) (Ci):

\[
\dot{D}_{\text{hourly}}(r) = \Gamma \frac{A}{r^{2}} e^{-\mu r} + D_{\text{bg}}
\]

\[
\dot{D}(r) = \frac{\dot{D}_{\text{hourly}}(r)}{3600},\qquad
D \leftarrow D + \dot{D}(r_t)\,\Delta t
\]

with default \(\Gamma = 1.32\) (order-of-magnitude Co-60 specific gamma-ray constant
in classical R·m²/(h·Ci) tables), \(\mu\) a small air attenuation, and \(r \ge 0.1\) m.

### Assumptions

1. **Point isotropic source** — no walls, shadowing, or self-absorption.
2. **No build-up factor** — pure exponential attenuation.
3. **Unit simplification** — \(\Gamma\) is historically roentgen-based. We treat the
   combined expression as a *dose-like* rate and label mission totals as “mSv” for
   threshold demos. **We do not claim perfect SI conversion from R to mSv.**
4. **\(D_{\text{bg}}\) is a lumped ambient term** in the same conversion path (legacy
   research consistency with the original project notebooks).
5. Default demo colocates the source with the valve (a *stressful* teaching case).

### What to use instead for real plants

- Shielded 3D dose maps, VISIPLAN-style ALARA planning, or Monte-Carlo transport (MCNP, etc.).
- This toolkit can later *import* a dose grid; v1 uses the analytic field above.

**Related literature:** Wright et al., *Simulating Ionising Radiation in Gazebo for
Robotic Nuclear Inspection Challenges*, Robotics 2021 (environment fidelity in sim).

---

## 3. Dynamics & lag

State:

\[
s = [x,\, y,\, \theta,\, v_{t-1},\, \omega_{t-1},\, v_{t-2},\, \omega_{t-2}]^\top
\]

Motion is driven by the **t−2** command (fixed FIFO lag of 2 steps).

### Assumptions

1. Flat plane unicycle kinematics.
2. **Fixed** lag depth (no random delay, reordering, or dropouts in v1).
3. No actuator saturation in the plant model.

**Context:** time delay is a classic teleoperation challenge; a fixed buffer is a
transparent teaching model (see surveys on telerobotic time delay).

---

## 4. Sensing & process noise

| Channel | Law | Role |
|---|---|---|
| Slip | \(\mathcal{N}(0,\sigma^2)\) i.i.d. on \(x,y\) each step | Process noise |
| Scramble | Bernoulli(\(p\)) on whether \((x,y)\) observation is replaced by a large uniform jump | Sensor corruption |

### Assumptions

1. Slip uncorrelated in time and between axes.
2. Scrambles affect **position only**, not heading, in v1.
3. Importance-sampling weights use the **joint** likelihood of slip + scramble.

---

## 5. Reachability

Axis-aligned **interval boxes**, linearized closed-loop map, Minkowski sum noise bounds
scaled like \(\sqrt{\Delta t}\).

### Modes

| Mode | Behavior |
|---|---|
| **`receding` (default)** | Every `receding_window` steps, re-seed a small box on the **nominal** state and re-propagate. Mitigates interval **wrapping / blow-up** for long missions. Runtime-assurance style operational assumption: re-cert is meaningful if the true state stayed near the plan. |
| **`open_loop`** | One tube for the whole horizon — often becomes vacuous (huge boxes) even when stochastic rollouts look fine. |
| **`both` / compare** | Primary receding result + open-loop companion metrics in the report. |

### Diagnostics

- **XY box area** \(w_x \cdot w_y\) over time (and peak **growth ratio** vs initial).
- **`blowup_suspected`** if peak growth ≥ threshold (default 50×).
- Not a Hamilton–Jacobi value function; not zonotopes. Those are future backends.

### Assumptions / limitations

1. Linearization can miss strong nonlinear effects.
2. Boxes over-approximate; open-loop tubes may be **vacuous** over long horizons.
3. Observation noise enters through the **control lag channel**, not by teleporting the true state.
4. Receding re-seeds assume proximity to the nominal path (not a full belief-state certificate).

**Context:** set-based reachability (JuliaReach, CORA); HJ reachability scales poorly on grids in high dimension (classic “curse of dimensionality”); neural HJ / CBF certificates are optional research extensions — not required for this toolkit’s default report.

---

## 6. Rare-event estimation

**Defensive mixture importance sampling:** sample from

\[
q = \alpha\, p_{\text{nom}} + (1-\alpha)\, p_{\text{bias}}
\]

with inflated slip \(\sigma\) and scramble \(p\), then reweight.

### Assumptions

1. Proposal family is correctly specified (Gaussian + Bernoulli).
2. Effective sample size (ESS) must be monitored — low ESS ⇒ unstable estimate.
3. Mission success CI uses the **nominal** subsample (true noise law).

**Context:** classical rare-event IS (e.g. L’Ecuyer surveys); defensive mixtures
for robust weights. Complementary tools: UQpy / OpenTURNS (general UQ),
Adaptive Stress Testing (find likely failures).

---

## 7. Success & failure definitions

| Name | Definition |
|---|---|
| **Task success** | Base enters radius `task_radius_m` of the valve |
| **Safety failure** | Cumulative dose \(> D_{\max}\) **or** max distance to nominal path \(> R_{\text{safe}}\) |
| **True mission success** | Reached valve **and** never safety-failed |

### Assumptions

1. No manipulation / handwheel torque model.
2. Corridor is defined w.r.t. the **nominal path**, not free-space CAD.
3. Empty plane — no collisions with walls.

---

## 8. Threshold rule (mission liveness)

We require the **lower end of the 95% Wald CI** on mission success to exceed
`min_mission_success` (default 90%). That is intentionally **conservative**.

ASSUMPTION: independent Bernoulli trials on the nominal subsample; Wald interval
(fine for large \(n\), weak for tiny \(n\)).

---

## 9. Visualizations

Plots colour the background by the **same** dose-rate field used in simulation
(log scale). Rollouts shown are a **subsample** for clarity (count stated in legend).
Reachability boxes are **x–y projections** of 7-D intervals.

---

## 10. How to cite / extend responsibly

When you publish numbers from this toolkit:

1. State the scenario YAML name + seed + `n_rollouts` + ESS.
2. Quote this assumptions list (or the JSON `assumptions` field).
3. Do **not** describe results as “certified safe for plant deployment.”
