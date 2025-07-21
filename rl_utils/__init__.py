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
from .environment import (
    cleanup_videos,
    create_env_with_wrappers,
    display_latest_video,
    make_vec_envs
)
from .networks import PolicyNetwork, ActorCriticNetwork
from .visualization import (
    get_moving_average,
    plot_training_results,
    plot_variance_analysis,
    plot_vectorized_training_results,
    plot_vectorized_variance_analysis,
    plot_rollout_based_training_results,
)

__all__ = [
    "create_env_with_wrappers",
    "make_vec_envs",
    "display_latest_video",
    "cleanup_videos",
    "plot_training_results",
    "plot_variance_analysis",
    "plot_vectorized_training_results",
    "plot_vectorized_variance_analysis",
    "plot_rollout_based_training_results",
    "get_moving_average",
    "PolicyNetwork",
    "ActorCriticNetwork",
    "set_seeds",
]
