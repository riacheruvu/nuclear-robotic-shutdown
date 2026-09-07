# remote_qual — Risk-Informed Qualification for Remote Robots

An open Python toolkit for qualifying autonomous and semi-autonomous mobile robots operating under **communication lag**, **sensor corruption**, and **radiation dose** constraints. Think of a robot approaching a manual valve for remote shutdown in a hazardous environment.

This toolkit takes a **scenario as input** and produces **qualification evidence as output**. It produces *research qualification evidence*, not a regulatory certification. If you're new to nuclear engineering or radiation units, I highly recommend starting with [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

There is also an interactive mission lab (browser) that you can check out. It features true 2-D lag dynamics, computed dose heatmap, presets, and Monte Carlo ghosts. It's meant for educational research viz:

```bash
python -m http.server 8000
# then open http://localhost:8000
```
Deployed at: [https://riacheruvu.github.io/nuclear-robotic-shutdown/](https://riacheruvu.github.io/nuclear-robotic-shutdown/)

---

## Plain-English Overview

Imagine a robot driving toward a valve in a radioactive room:
1. **Lag**: the command you send does not move the wheels instantly (teleoperation delay).
2. **Noise**: wheels slip; sometimes the position sensor completely scrambles.
3. **Dose**: the longer the robot sits near a hot source, the more dose it accumulates.

This toolkit answers three main questions:

| Question | Method |
|---|---|
| Could uncertainty push the robot out of a safe corridor? | Linearized **interval reachability** |
| How often do safety failures happen? | **Defensive mixture importance sampling** (rare-event MC) |
| How often does the mission actually succeed? | Success rate + 95% confidence interval |

It writes a clear **JSON + Markdown report** and an optional plot with the dose field and reachability boxes.

---

## Quick Start

Dependencies are intentionally kept simple (numpy, PyTorch, PyYAML, and matplotlib for visualization).

```bash
# Clone the repository
git clone https://github.com/riacheruvu/nuclear-robotic-shutdown.git
cd nuclear-robotic-shutdown

# Install the toolkit
pip install -e ".[viz,dev]"
```

Here are some examples of running scenarios:

```bash
# 1. Validate a scenario configuration without running it
remote-qual validate scenarios/valve_baseline.yaml

# 2. Run a scenario with 2000 rollouts
remote-qual run scenarios/valve_baseline.yaml --rollouts 2000

# 3. Run a high-slip scenario and save an animation of the rollout
remote-qual run scenarios/high_slip.yaml --animation

# 4. Run an ablation study to compare Monte Carlo against Defensive Mixture
remote-qual run scenarios/valve_baseline.yaml --ablation --rollouts 2000
```

Outputs will land in `out/<scenario_name>/`:
- `*_report.json` — machine-readable evidence
- `*_report.md` — plain-English summary
- `*_snapshot.png` — dose field, rollouts, and reachability boxes

---

## What is in a report?

- Mission success rate and 95% CI
- Safety failure probability with standard error and **effective sample size (ESS)**
- Formal reachability flag (`reachability_unsafe`)
- Explicit list of assumptions
- Methods used

---

## Repository Layout

```text
scenarios/                 # YAML scenario library
src/remote_qual/
  scenario/                # Loader and schema
  plugins/                 # Swappable dynamics, controllers, noise, etc.
  verification/            # Reachability & Importance Sampling math
  report/                  # JSON/MD generation
  viz/                     # Plotting
docs/
  ASSUMPTIONS.md           # Important: read this first!
tests/                     # Test suite
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) if you'd like to help.

---

## Starter Scenarios

| Scenario | Focus |
|---|---|
| `valve_baseline` | Default setup |
| `high_scramble` | Sensor integrity |
| `high_slip` | Corridor tracking |
| `high_activity` | Dose accumulation |
| `stress_combined` | All of the above |

---

## Scientific Scope

**Included (v0.1):**
- 7-D unicycle with fixed 2-step command lag
- Proportional heading controller
- Gaussian slip & Bernoulli scrambles
- Simplified point-source dose field
- Linearized axis-aligned interval reachability
- Defensive mixture importance sampling

**Not included:**
- Full plant CAD / shielding transport
- Random network jitter
- Manipulation

---

## Citation & Provenance

This codebase builds upon the framework discussed in my [Medium article](https://riacheruvu.medium.com/risk-informed-robotic-shutdown-for-nuclear-engineering-1c154c72fbaa) and represents independent research.

This work was accepted for presentation at the **2026 American Nuclear Society Winter Conference** (Robotics and Remote Systems Division) in Phoenix, AZ:

> Cheruvu, R. (2026). *Risk-Informed Robotic Remote Shutdown: Combining Reachability Envelopes and Importance Sampling*. 2026 American Nuclear Society Winter Conference & Expo. [Session Link](https://www.ans.org/meetings/wc2026/session/view-4015/).

## License

MIT — see [LICENSE](LICENSE).
