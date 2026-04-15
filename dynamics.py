"""
dynamics.py — Module 0: POMDP Dynamics & Controller

Augmented 7-D state:  s = [x, y, θ, v_{t-1}, ω_{t-1}, v_{t-2}, ω_{t-2}]

The two-step lag buffer means the kinematics at time t are driven by the
command issued at t-2, which captures realistic actuation delay.
"""

import numpy as np
from config import DT, K_P, V_NOMINAL


# ── Low-level controller ──────────────────────────────────────────────────────

def get_control(s_obs: np.ndarray) -> np.ndarray:
    """Proportional heading controller that steers toward the origin (valve).

    Args:
        s_obs: Observed state vector (7-D); only first 3 elements are used.

    Returns:
        u = [v, ω]  — forward speed and angular velocity commands.
    """
    x, y, theta = s_obs[:3]
    desired_heading = np.arctan2(-y, -x)
    heading_error = desired_heading - theta
    # Wrap to [-π, π]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    omega = K_P * heading_error
    v = V_NOMINAL * (1.0 - 0.5 * abs(heading_error) / np.pi)
    return np.array([v, omega])


# ── Deterministic dynamics ────────────────────────────────────────────────────

def nominal_dynamics(s: np.ndarray) -> np.ndarray:
    """One deterministic step under temporal lag (u_{t-2} drives kinematics).

    Args:
        s: Current 7-D state.

    Returns:
        s_next: Next 7-D state (no noise).
    """
    u_new = get_control(s)

    s_next = s.copy()
    v_lagged, omega_lagged = s[5], s[6]

    s_next[0] += v_lagged * np.cos(s[2]) * DT
    s_next[1] += v_lagged * np.sin(s[2]) * DT
    s_next[2] += omega_lagged * DT

    # Shift lag buffer: (t-1) → (t-2), new command → (t-1)
    s_next[5:7] = s[3:5]
    s_next[3:5] = u_new

    return s_next


def compute_nominal_trajectory(s0_3d: np.ndarray, T: int) -> np.ndarray:
    """Roll out the noise-free trajectory from a 3-D starting pose.

    Args:
        s0_3d: Initial [x, y, θ].
        T:     Maximum number of steps.

    Returns:
        Array of shape (T_actual+1, 7).
    """
    s = np.concatenate([s0_3d, np.zeros(4)])  # zero-initialise lag buffers
    traj = [s.copy()]
    for _ in range(T):
        s = nominal_dynamics(s)
        traj.append(s.copy())
        if np.linalg.norm(s[:2]) < 0.15:  # φ_task success radius
            break
    return np.array(traj)


# ── Linearisation ─────────────────────────────────────────────────────────────

def jacobian_dynamics(s: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numerical Jacobian A = ∂f/∂s of nominal_dynamics at state s.

    Args:
        s:   Operating-point state (7-D).
        eps: Finite-difference step size.

    Returns:
        A: (7, 7) Jacobian matrix.
    """
    n = len(s)
    f0 = nominal_dynamics(s)
    A = np.zeros((n, n))
    for i in range(n):
        s_plus = s.copy()
        s_plus[i] += eps
        A[:, i] = (nominal_dynamics(s_plus) - f0) / eps
    return A
