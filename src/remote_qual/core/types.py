"""Shared type aliases for readability."""

from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt

# [x, y, theta]
Pose3 = npt.NDArray[np.floating]

# [x, y, theta, v_{t-1}, omega_{t-1}, v_{t-2}, omega_{t-2}]
State7 = npt.NDArray[np.floating]

# [v, omega] forward speed (m/s) and yaw rate (rad/s)
Control = npt.NDArray[np.floating]

ArrayLike = Union[np.ndarray, list, tuple]
