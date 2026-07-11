# remote_qual — Risk-Informed Qualification for Remote Robots

**Scenario in → evidence out.**

An open Python toolkit for qualifying autonomous / semi-autonomous mobile robots that must operate under **communication lag**, **sensor corruption**, and **radiation dose** constraints — for example, approaching a manual valve for remote shutdown.

> **Research evidence, not regulatory certification.**  
> Every report lists assumptions. Read [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) if you are new to nuclear engineering or radiation units.

**Interactive mission lab (browser):** open [`index.html`](index.html) locally or on GitHub Pages —  
true 2-D lag dynamics, computed dose heatmap, presets, Monte Carlo ghosts.  
Still educational research viz (not the Python report pipeline).

```bash
python -m http.server 8000
# then http://localhost:8000
```

Deployed: [https://riacheruvu.github.io/nuclear-robotic-shutdown/](https://riacheruvu.github.io/nuclear-robotic-shutdown/)

Write-up: [Modeling Safety-Critical Robots for Nuclear Engineering](https://riacheruvu.medium.com/risk-informed-robotic-shutdown-for-nuclear-engineering-1c154c72fbaa)

---

## Plain-English overview

Imagine a robot driving toward a valve in a radioactive room:

1. **Lag** — the command you send does not move the wheels instantly (teleoperation delay).
2. **Noise** — wheels slip; sometimes the position sensor “scrambles.”
3. **Dose** — the longer the robot sits near a hot source, the more dose accumulates.

This toolkit answers:

| Question | Method |
|---|---|
| Could uncertainty push the robot out of a safe corridor? | Linearized **interval reachability** |
| How often do safety failures happen? | **Defensive mixture importance sampling** (rare-event MC) |
| How often does the mission actually succeed? | Success rate + 95% confidence interval |

It writes a **JSON + Markdown report** and an optional scientifically tied plot (dose field from the same formula used in simulation).

---

## Quick start

Dependencies are intentionally **few and mainstream** (numpy, PyTorch, PyYAML;
matplotlib optional). See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

```bash
git clone https://github.com/riacheruvu/nuclear-robotic-shutdown.git
cd nuclear-robotic-shutdown
pip install -e ".[viz,dev]"

# Validate a scenario
remote-qual validate scenarios/valve_baseline.yaml

# Run qualification (2000 rollouts by default in YAML; increase for paper-style runs)
remote-qual run scenarios/valve_baseline.yaml --rollouts 2000

# List swappable plugins
remote-qual list-plugins
```

Outputs land in `out/<scenario_name>/`:

- `*_report.json` — machine-readable evidence  
- `*_report.md` — plain-English summary  
- `*_snapshot.png` — dose field + rollouts + reachability boxes  

Legacy:

```bash
python main.py
```

---

## What is in a report?

- Mission success rate and 95% CI (conservative rule: lower CI bound vs threshold)
- \(P(\text{fail})\) with standard error and **effective sample size (ESS)**
- Formal reachability flag (`reachability_unsafe`)
- Explicit **assumptions** list
- Methods used (dynamics, noise, estimator parameters)

---

## Repository layout

```text
scenarios/                 # user-facing experiment library (YAML)
src/remote_qual/
  scenario/                # loader + schema
  plugins/                 # dynamics, controller, noise, hazard, failure
  verification/            # reachability + rare-event IS
  report/                  # JSON/MD evidence
  viz/                     # scientifically tied plots
docs/
  ASSUMPTIONS.md           # start here if new to nuclear eng
  architecture.md
tests/
index.html                 # browser dashboard (supplementary)
```

See [`docs/architecture.md`](docs/architecture.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Starter scenarios

| Scenario | Stresses |
|---|---|
| `valve_baseline` | Default demo pose / noise |
| `high_scramble` | Sensor integrity |
| `high_slip` | Corridor tracking |
| `high_activity` | Dose accumulation |
| `stress_combined` | Everything harder (+ ablation) |

---

## Scientific scope (honest summary)

**Included (v0.1)**

- 7-D unicycle with **fixed 2-step command lag**
- Proportional heading controller
- Gaussian slip + Bernoulli salt-and-pepper scrambles
- Simplified **point-source** dose field (Co-60-like \(\Gamma\))
- Linearized axis-aligned **interval reachability**
- **Defensive mixture** importance sampling with joint slip+scramble weights

**Not included (yet)**

- Full plant CAD / shielding / MCNP transport  
- Spot / ROS bridges  
- Random network jitter  
- Manipulation / turning the handwheel  

Details: [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

### Related tools (complementary, not competitors)

| Tool / ecosystem | Role |
|---|---|
| Spot SDK + rad mapping payloads | Real platforms & surveys |
| Gazebo radiation plugins, RCI_radiation | High-fidelity radiation-aware nav sim |
| VISIPLAN / DEMplus | Industry ALARA / decommissioning planning |
| JuliaReach, CORA, PyBDR | General set-based reachability backends |
| UQpy, OpenTURNS | General UQ / reliability |
| Scenic, VerifAI, Adaptive Stress Testing | Scenario languages & falsification |

**Our niche:** open **scenario → formal + rare-event qualification report** for latency-aware remote hazardous tasks.

---

## Citation / provenance

This repository implements and productizes the framework discussed in the Medium article above (AA228V / CS238V lineage). Dose-unit conventions and the joint-likelihood IS fix from the original research code are preserved and documented.

---

## License

MIT — see [LICENSE](LICENSE).
