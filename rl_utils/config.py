import random

import numpy as np
import torch


def set_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
