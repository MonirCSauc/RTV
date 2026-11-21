"""One-Euro filter (Casiez et al., 2012): a low-pass filter whose cutoff rises with speed.

A fixed low-pass filter has to choose between jitter (weak smoothing) and lag (strong smoothing). The One-Euro
filter smooths strongly while the signal is nearly still, where jitter is visible, and weakly while it moves fast,
where lag is visible. Two parameters: `min_cutoff` (Hz; lower = steadier at rest) and `beta` (higher = less lag
in motion).
"""
import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters a numpy array of any shape, element by element, with one speed estimate per element.

    group_axis: if given, the speed is the vector norm along that axis (e.g. -1 for an array of 3D points or
    quaternions), so all components of one point share one cutoff and the point does not move along a bent path.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, group_axis=None):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.group_axis = group_axis
        self.x = None
        self.dx = None

    def reset(self):
        self.x = None
        self.dx = None

    def __call__(self, x, dt):
        x = np.asarray(x, np.float64)
        if self.x is None or dt <= 0:
            self.x = x.copy()
            self.dx = np.zeros_like(x)
            return x.astype(np.float32)
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        speed = (np.abs(self.dx) if self.group_axis is None
                 else np.linalg.norm(self.dx, axis=self.group_axis, keepdims=True))
        cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x = self.x + a * (x - self.x)
        return self.x.astype(np.float32)
