"""
Gaussian position slip + Bernoulli salt-and-pepper sensor scrambles.

Plain English
-------------
Two kinds of randomness:

1. **Slip** — every step, the true (x, y) position jitters a little
   (wheels slide, floor is uneven). Modelled as independent Gaussians.
2. **Scramble** — occasionally the position sensor reports total nonsense
   (think radiation-induced bit flips or a hard glitch). With probability p,
   (x, y) observations jump uniformly in a box.

Importance-sampling note
------------------------
Defensive mixture IS biases *both* slip variance and scramble probability,
and the likelihood weight must account for **both** (joint likelihood).
That prevents weight collapse when only one noise channel is corrected.

ASSUMPTIONS
-----------
1. Slip is i.i.d. isotropic Gaussian in x and y each step (no correlation).
2. Scrambles are i.i.d. Bernoulli; when they fire, observation noise is
   uniform on [-scramble_xy_range, +scramble_xy_range]² (not used in the
   likelihood of the scramble *indicator*, only the Bernoulli part is).
3. Heading θ is not directly scrambled in v1 (only x, y).
"""

from __future__ import annotations

import torch

from remote_qual.plugins.base import NoisePlugin


class SlipScrambleNoise(NoisePlugin):
    def __init__(
        self,
        sigma_slip: float = 0.05,
        p_scramble: float = 0.05,
        scramble_xy_range: float = 2.0,
    ):
        self.sigma_slip = float(sigma_slip)
        self.p_scramble = float(p_scramble)
        self.scramble_xy_range = float(scramble_xy_range)

    def sample_slips(self, T: int, B: int, device: str, bias_factor: float = 1.0):
        sigma = self.sigma_slip * bias_factor
        return torch.normal(mean=0.0, std=sigma, size=(T, B, 2), device=device)

    def sample_scrambles(self, T: int, B: int, device: str, bias_factor: float = 1.0):
        p = min(self.p_scramble * bias_factor, 0.9)
        return (torch.rand(T, B, device=device) < p).float()

    def biased_scramble_prob(self, bias_factor: float) -> float:
        return min(self.p_scramble * bias_factor, 0.9)

    def log_prob_joint(self, slips, scrambles, *, biased: bool, bias_factor: float):
        if biased:
            sigma = self.sigma_slip * bias_factor
            p = self.biased_scramble_prob(bias_factor)
        else:
            sigma = self.sigma_slip
            p = self.p_scramble
        dist_slip = torch.distributions.Normal(0.0, sigma)
        dist_scram = torch.distributions.Bernoulli(probs=p)
        log_slip = dist_slip.log_prob(slips).sum(dim=(0, 2))
        log_scram = dist_scram.log_prob(scrambles).sum(dim=0)
        return log_slip + log_scram
