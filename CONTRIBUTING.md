# Contributing to `remote_qual`

Thanks for helping build an open **risk-informed qualification** toolkit for
remote robots in hazardous environments.

## Ground rules

1. **State assumptions.** If you change physics or statistics, update
   `docs/ASSUMPTIONS.md` and the report’s `assumptions` list.
2. **Keep numbers honest.** No silent unit changes. Prefer tests that lock
   conventions (especially dose rate conversion).
3. **Plain English welcome.** Docstrings should be readable by novices.
4. **Research ≠ certification.** Do not market outputs as regulatory approval.

## Quick dev setup

```bash
pip install -e ".[dev,viz]"
remote-qual validate scenarios/valve_baseline.yaml
pytest -q
```

## Add a scenario (easiest contribution)

1. Copy `scenarios/valve_baseline.yaml`.
2. Change `name`, `seed`, and the parameters you care about.
3. Run:

```bash
remote-qual run scenarios/your_scenario.yaml --rollouts 500
```

4. Open the JSON/MD report under `out/your_scenario/`.

## Add a plugin

1. Implement the interface in `src/remote_qual/plugins/base.py`.
2. Register it in `src/remote_qual/plugins/registry.py`.
3. Add a tiny unit test.
4. Reference the plugin name from a scenario YAML.

## Code style

- Python 3.9+.
- Prefer clear names over cleverness.
- Keep optional heavy deps (`matplotlib`, `ffmpeg`) out of the core import path
  when possible.

## Dependencies & external downloads

Be conservative. New libraries or anything fetched from the network must be
**needed, mainstream/maintained, and license-compatible**. Prefer the stdlib and
existing stack (`numpy`, `torch`, `PyYAML`, optional `matplotlib`).

Full policy and current inventory: [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

- Do **not** add drive-by dependencies for convenience.  
- Use `yaml.safe_load` (already required) — never `yaml.load` without a loader.  
- Optional features (plots, linters) stay in extras, not core.

## Related tools (please interoperate, don’t duplicate)

| Ecosystem | Use when… |
|---|---|
| Gazebo + radiation plugins / RCI_radiation | High-fidelity env + Nav2 |
| Spot SDK | Real platform deployment |
| JuliaReach / CORA / PyBDR | Heavier set-based reachability backends |
| UQpy / OpenTURNS | General UQ & reliability |
| Scenic / VerifAI / AST | Scenario languages & falsification |

This project’s niche is **scenario → formal + rare-event qualification report**
for latency-aware remote hazardous tasks.
