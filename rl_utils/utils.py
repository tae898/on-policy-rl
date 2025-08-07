import random

import numpy as np
import torch


def set_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class RunningMeanStd:
    """Tracks running mean and variance (Welford)."""

    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps  # avoids div-by-zero the first update

    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        delta = batch_mean - self.mean
        tot_ct = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_ct
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_ct
        new_var = M2 / tot_ct

        self.mean, self.var, self.count = new_mean, new_var, tot_ct
