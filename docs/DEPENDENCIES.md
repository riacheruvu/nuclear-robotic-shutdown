# Dependencies policy

We keep the external surface **small, mainstream, and justified**.

New packages (or any content downloaded from the internet — wheels, models,
datasets, JS CDNs in the dashboard) should be:

1. **Needed** — solves a real problem we cannot cover with stdlib + existing deps  
2. **Trusted** — widely used, actively maintained, clear provenance (prefer PyPI
   projects with known orgs / long history)  
3. **Licensed** — compatible with MIT (or documented if not)  
4. **Reviewed** — someone reads what it does; no “mystery one-liner installs”

If in doubt: **do not add it**. Discuss in an issue first.

---

## Runtime dependencies (core)

| Package | Why we need it | Notes |
|---|---|---|
| **numpy** | Arrays, kinematics, numerics | De-facto scientific Python standard |
| **torch** | Batched rollouts / IS on CPU or GPU | Heavy but already central to the rare-event engine; prefer official PyTorch installs |
| **PyYAML** | Scenario YAML loading | Standard, small; we use `safe_load` only |

## Optional

| Extra | Packages | Why |
|---|---|---|
| `viz` | **matplotlib** | Static plots / animation; not required to compute reports |
| `dev` | **pytest**, **ruff**, matplotlib | Tests and lint only |

## Explicitly avoided (for now)

| Tempting addition | Why we skip it |
|---|---|
| Full ROS / Gazebo stacks | Huge surface; optional bridge later, not core |
| Random small GitHub-only libs | Prefer stdlib or the packages above |
| Unpinned “latest everything” in production docs | We use minimum lower bounds; lockfiles welcome later for CI reproducibility |

---

## Install guidance (secure defaults)

```bash
# Prefer the project definition over ad-hoc pip installs of unknowns
pip install -e ".[viz,dev]"

# Or the pinned-ish requirements list (same known packages)
pip install -r requirements.txt
```

- Install from **PyPI** (or your org’s mirror), not unknown zip URLs.  
- For **PyTorch**, follow [pytorch.org](https://pytorch.org) if you need a
  specific CUDA build — still the same upstream project.  
- The browser `index.html` dashboard should only load scripts from **known CDNs**
  (e.g. KaTeX) if any; prefer vendoring if supply-chain risk is a concern.

---

## Adding a dependency (checklist)

- [ ] Can stdlib / numpy / torch do this already?  
- [ ] Is the package popular and maintained (recent releases, issue activity)?  
- [ ] License OK?  
- [ ] Documented in this file + `pyproject.toml` / `requirements.txt`?  
- [ ] Optional extra vs core? Prefer **optional** when possible.  
- [ ] Tests still pass offline (no surprise network calls at import time)?  

**Rule of thumb:** core install stays lean so students and labs can trust the stack.
