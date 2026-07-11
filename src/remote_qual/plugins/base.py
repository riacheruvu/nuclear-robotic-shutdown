"""Abstract plugin interfaces (protocols + light base classes)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class DynamicsPlugin(ABC):
    """Plant model: how state evolves given a control command."""

    state_dim: int = 7

    @abstractmethod
    def augment_initial(self, pose3: np.ndarray) -> np.ndarray:
        """Lift [x, y, θ] into the full state (e.g. zero lag buffers)."""

    @abstractmethod
    def step(self, s: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """One deterministic kinematic step (noise applied elsewhere)."""

    @abstractmethod
    def jacobian(self, s: np.ndarray) -> np.ndarray:
        """∂f/∂s at state s for linearized reachability."""


class ControllerPlugin(ABC):
    """Maps an *observation* of state to a control command u = [v, ω]."""

    @abstractmethod
    def __call__(self, s_obs: np.ndarray) -> np.ndarray:
        ...


class NoisePlugin(ABC):
    """Process / sensor noise used in stochastic rollouts and IS weights."""

    @abstractmethod
    def sample_slips(self, T: int, B: int, device: str, bias_factor: float = 1.0):
        """Return (T, B, 2) slip samples."""

    @abstractmethod
    def sample_scrambles(self, T: int, B: int, device: str, bias_factor: float = 1.0):
        """Return (T, B) Bernoulli scramble indicators (float 0/1)."""

    @abstractmethod
    def log_prob_joint(self, slips, scrambles, *, biased: bool, bias_factor: float):
        """Joint log-density under nominal (biased=False) or biased proposal."""


class HazardPlugin(ABC):
    """Environment risk: dose field + task geometry."""

    @abstractmethod
    def dose_rate(self, xy) -> Any:
        """Dose-like rate per second at positions xy (..., 2)."""

    @abstractmethod
    def distance_to_valve(self, xy) -> Any:
        ...

    @abstractmethod
    def task_reached(self, xy, task_radius: float) -> Any:
        ...


class FailurePlugin(ABC):
    """Maps trajectories + cumulative dose to a boolean failure mask."""

    @abstractmethod
    def __call__(
        self,
        trajs,
        doses,
        nominal_path: np.ndarray,
        *,
        corridor_radius: float,
        d_max: float,
        device: str,
    ):
        ...
