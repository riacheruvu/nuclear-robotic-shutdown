# Architecture (v0.1)

```text
Scenario YAML  →  Plugin registry  →  Pipeline  →  Report (JSON/MD) + plots
                         │
         ┌───────────────┼────────────────┐
         ▼               ▼                ▼
    Dynamics        Rare-event IS    Interval reachability
    Controller      (defensive mix)  (linearized boxes)
    Noise / Hazard
```

## Package map

| Path | Role |
|---|---|
| `src/remote_qual/scenario/` | YAML load + validation |
| `src/remote_qual/plugins/` | Swappable domain models |
| `src/remote_qual/verification/` | Formal + statistical engines |
| `src/remote_qual/report/` | Evidence artifacts |
| `src/remote_qual/viz/` | Scientifically tied figures |
| `scenarios/` | User-facing experiment library |

## Reachability modes

Default formal engine: linearized **interval boxes** with **receding-horizon
re-certification** (re-seed every `receding_window` steps on the nominal).
Reports include **XY box-area growth** diagnostics and an optional open-loop
companion (to illustrate interval blow-up). This is not Hamilton–Jacobi; see
`docs/ASSUMPTIONS.md`.

## Extension points

1. New scenario → add YAML under `scenarios/`.
2. New controller/noise/hazard → new file under `plugins/` + registry entry.
3. New estimator → `verification/rare_events/` + method name in YAML.
4. Future: zonotope / reduced-order HJ backends under `verification/reachability/`.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
