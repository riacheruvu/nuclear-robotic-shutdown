"""Export qualification reports to JSON (and optional Markdown)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from remote_qual.report.model import QualificationReport

PathLike = Union[str, Path]


def export_report(report: QualificationReport, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


def export_markdown_summary(report: QualificationReport, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = report.metrics
    v = report.thresholds.get("verdict", {})
    lines = [
        f"# Qualification report: {report.scenario_name}",
        "",
        report.plain_english_summary,
        "",
        "## Metrics",
        f"- Mission success: {m.get('mission_success_rate', float('nan')):.1%} "
        f"± {m.get('mission_success_ci95_halfwidth', float('nan')):.1%} (95% CI half-width)",
        f"- P(fail): {m.get('p_fail', float('nan')):.6f} ± {m.get('p_fail_std', float('nan')):.6f}",
        f"- ESS: {m.get('ess', float('nan')):.1f} / {m.get('n_rollouts', '?')}",
        f"- Reachability unsafe: {m.get('reachability_unsafe')}",
        "",
        f"## Verdict: **{v.get('overall', 'n/a').upper()}**",
        f"- Mission liveness: {v.get('mission_liveness')}",
        f"- P(fail) threshold: {v.get('p_fail')}",
        "",
        f"> {report.disclaimer}",
        "",
        "## Key assumptions",
    ]
    for a in report.assumptions:
        lines.append(f"- {a}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
