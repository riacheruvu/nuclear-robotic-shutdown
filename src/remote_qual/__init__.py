"""
remote_qual — Risk-informed qualification for remote robots.

Plain-English idea
------------------
Remote robots (for example, approaching a valve in a high-radiation room)
face uncertainty: communication lag, slippery wheels, noisy sensors, and
dose accumulating over time. This toolkit turns a *scenario* into *evidence*:

1. Formal reachability bounds (where could the robot be in the worst case?)
2. Rare-event statistics (how often do safety or task failures happen?)

Disclaimer
----------
Outputs are **research qualification evidence**, not a regulatory safety
certificate. Always read the assumptions printed in each report.
"""

from remote_qual._version import __version__
from remote_qual.pipeline import run_qualification
from remote_qual.report.model import QualificationReport

__all__ = ["__version__", "run_qualification", "QualificationReport"]
