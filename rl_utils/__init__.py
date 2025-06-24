"""
RL Learning Series Utilities

A utility package containing shared components for the RL learning series:
- Environment wrappers and preprocessing
- Visualization and plotting functions
- Common neural network architectures
- Video recording and testing utilities

This package separates infrastructure code from algorithm-specific learning content.
"""

__version__ = "1.0.0"

from .config import set_seeds
from .environment import (cleanup_videos, create_env_with_wrappers,
                          display_latest_video, preprocess_state)
from .networks import PolicyNetwork
from .visualization import (get_moving_average, plot_training_results,
                            plot_variance_analysis)

__all__ = [
    "preprocess_state",
    "create_env_with_wrappers",
    "display_latest_video", 
    "cleanup_videos",
    "plot_training_results",
    "plot_variance_analysis",
    "get_moving_average",
    "PolicyNetwork",
    "set_seeds"
]
