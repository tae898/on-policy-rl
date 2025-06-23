import random

import numpy as np
import torch


def create_base_config():
    """Create base configuration for RL experiments."""
    return {
        "seed": 42,
        "episodes": 1000,
        "gamma": 0.99,
        "lr": 1e-4,
        "device": "cpu",
        "scores_window_size": 100,

        # Environment: LunarLander-v3 only
        "env_id": "LunarLander-v3",
        "env_kwargs": {
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        },

        # Video Recording Config
        "record_videos": True,
        "video_folder": "videos",
        "video_record_interval": 200,
        "record_test_videos": True,

        # Neural Network Config
        "policy_network": {
            "fc_out_features": [64, 32],
        }
    }

def set_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
