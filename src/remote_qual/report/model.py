"""Qualification report data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


DISCLAIMER = (
    "Research qualification evidence for scientific exploration — "
    "not a regulatory certification. Read assumptions in the toolkit docs "
    "(dose unit simplification, point-source radiation, fixed lag, empty plane)."
)


@dataclass
class QualificationReport:
    schema_version: str = "1.0"
    toolkit_version: str = "0.1.0"
    scenario_name: str = ""
    seed: int = 0
    timestamp_utc: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    ablation: Optional[Dict[str, Any]] = None
    disclaimer: str = DISCLAIMER
    plain_english_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
