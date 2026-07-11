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

## Extension points

1. New scenario → add YAML under `scenarios/`.
2. New controller/noise/hazard → new file under `plugins/` + registry entry.
3. New estimator → `verification/rare_events/` + method name in YAML.

See [ASSUMPTIONS.md](ASSUMPTIONS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
