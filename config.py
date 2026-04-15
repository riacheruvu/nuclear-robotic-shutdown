"""
config.py — Shared system & environment constants.

A Risk-Informed Qualification Framework for Robotic Remote Shutdown
AA228V / CS238V Final Project
"""

import numpy as np

# ── Robot / Task ──────────────────────────────────────────────────────────────
VALVE_POS  = np.array([0.0, 0.0])
R_SAFE     = 0.5    # max corridor deviation (m)
D_MAX      = 50.0   # max cumulative dose (mSv)
T_HORIZON  = 100    # episode length (steps)
DT         = 0.1    # time step (s)

# ── FIX: Tuned Controller Gains ───────────────────────────────────────────────
# Increased K_P for aggressive recovery from salt-and-pepper scrambles.
# Increased V_NOMINAL to minimize residence time in the high-dose gradient.
K_P        = 2.5    # proportional heading gain (tuned from 1.5)
V_NOMINAL  = 0.5    # nominal forward speed (m/s) (tuned from 0.3)

# ── Radiation / Environment ───────────────────────────────────────────────────
# GAMMA_CONST is in R·m²/(hr·Ci).  Divide by 3600 when multiplying by DT (s)
# so units are consistent: [R·m²/(hr·Ci)] * [Ci] / [m²] / 3600 → [R/s].
GAMMA_CONST = 1.32    # Co-60 specific gamma-ray constant [R·m²/(hr·Ci)]
ACTIVITY    = 15.0    # source activity [Ci]
MU_AIR      = 0.0001  # air attenuation coefficient [1/m]
D_BG        = 0.05    # ambient background dose rate [mSv/s]

# ── Sensor Noise ──────────────────────────────────────────────────────────────
P_SALT_PEP  = 0.05   # probability of full position-sensor scramble per step