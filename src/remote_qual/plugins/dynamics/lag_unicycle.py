"""
7-D unicycle with a two-step command lag buffer.

Plain English
-------------
When you teleoperate a robot over a network, the command you send now may
not move the wheels until a short time later. We model that with a FIFO
buffer: the motion at time t is driven by the command issued at t-2.

State layout
------------
    s = [x, y, θ, v_{t-1}, ω_{t-1}, v_{t-2}, ω_{t-2}]

Kinematics (discrete, Δt seconds)
---------------------------------
    x ← x + v_{t-2} · cos(θ) · Δt
    y ← y + v_{t-2} · sin(θ) · Δt
    θ ← θ + ω_{t-2} · Δt
    then shift the lag buffer and insert the new command u = [v, ω].

ASSUMPTIONS
-----------
1. Fixed lag depth of ``lag_steps`` (default 2). No random jitter/packet loss
   in v1 (those are natural extensions).
2. Perfect unicycle kinematics on a flat plane (no wheel slip in the
   deterministic map; slip is added by the noise plugin).
3. No actuator saturation unless a future controller/dynamics plugin adds it.

Literature context
------------------
Delayed / buffered teleoperation is a classical challenge in telerobotics
(see surveys on time-delay teleoperation). Fixed-step lag is a transparent
teaching model used in this framework's original AA228V project formulation.
"""

from __future__ import annotations

import numpy as np

from remote_qual.plugins.base import DynamicsPlugin


class LagUnicycleDynamics(DynamicsPlugin):
    """Unicycle plant with a FIFO control lag buffer."""

    state_dim = 7

    def __init__(self, lag_steps: int = 2):
        if lag_steps != 2:
            # v1 implements the validated 2-step buffer layout.
            # Other depths need an explicit state layout redesign.
            raise ValueError(
                "v1 only supports lag_steps=2 (state layout is fixed 7-D). "
                "Requested lag_steps=%r" % (lag_steps,)
            )
        self.lag_steps = lag_steps

    def augment_initial(self, pose3: np.ndarray) -> np.ndarray:
        pose3 = np.asarray(pose3, dtype=float).reshape(3)
        return np.concatenate([pose3, np.zeros(4)])

    def step(self, s: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        s = np.asarray(s, dtype=float).reshape(7)
        u = np.asarray(u, dtype=float).reshape(2)
        s_next = s.copy()
        v_lagged, omega_lagged = s[5], s[6]
        s_next[0] += v_lagged * np.cos(s[2]) * dt
        s_next[1] += v_lagged * np.sin(s[2]) * dt
        s_next[2] += omega_lagged * dt
        # Shift lag buffer: (t-1) → (t-2), new command → (t-1)
        s_next[5:7] = s[3:5]
        s_next[3:5] = u
        return s_next

    def jacobian(self, s: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numerical Jacobian of the closed-loop step is provided by the
        pipeline via nominal dynamics that includes the controller.

        Here we expose open-loop ∂f/∂s with zero new command (identity-ish
        lag shift). Prefer ``closed_loop_jacobian`` in the controller-aware
        helper used by reachability.
        """
        # Open-loop with u = current lag slot as proxy; reachability uses
        # the dedicated closed-loop routine in verification.
        u = s[3:5].copy()
        f0 = self.step(s, u, dt=0.1)
        n = self.state_dim
        A = np.zeros((n, n))
        for i in range(n):
            s_plus = s.copy()
            s_plus[i] += eps
            A[:, i] = (self.step(s_plus, u, dt=0.1) - f0) / eps
        return A
