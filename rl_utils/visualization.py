import matplotlib.pyplot as plt
import numpy as np


def get_moving_average(values, window=100):
    """Calculate moving average for smoothing noisy data."""
    if len(values) < window:
        return np.array(values)

    # Use np.convolve with 'valid' mode
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")

    # Create corresponding x-axis values (offset by window-1)
    x_offset = window - 1
    return smoothed, x_offset


def plot_training_scores(scores, config, action_type, algorithm_name="Algorithm"):
    """Plot training scores with moving average."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    fig.suptitle(f'{algorithm_name} ({action_type}) Scores on {config["env_id"]}')

    # Plot scores
    ax.plot(
        scores,
        label="Raw Score",
        alpha=0.3,
        color="blue" if action_type == "Discrete" else "red",
    )

    # Handle moving average properly
    if len(scores) >= config["window_length"]:
        smoothed, x_offset = get_moving_average(
            scores, window=config["window_length"]
        )
        smoothed_episodes = range(x_offset + 1, x_offset + 1 + len(smoothed))
        ax.plot(
            smoothed_episodes,
            smoothed,
            label=f'Smoothed Score ({config["window_length"]}ep)',
            color="blue" if action_type == "Discrete" else "red",
            linewidth=2,
        )

    # Use configurable target score
    ax.axhline(
        y=config["target_score"],
        color="g",
        linestyle="--",
        label=f'Target Score ({config["target_score"]})',
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Episode")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_training_losses(losses_dict, config, action_type, algorithm_name="Algorithm"):
    """
    Plot training losses with all components in a single plot with different colors.

    Args:
        losses_dict: Dictionary with loss components, e.g.:
                    {'policy_loss': [...], 'value_loss': [...], 'entropy_loss': [...], 'total_loss': [...]}
                    For REINFORCE: {'policy_loss': [...], 'total_loss': [...]} (where total == policy)
        config: Configuration dictionary
        action_type: "Discrete" or "Continuous"
        algorithm_name: Name of the algorithm
    """
    loss_components = list(losses_dict.keys())

    if len(loss_components) == 0:
        print("No loss data to plot")
        return

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    fig.suptitle(f'{algorithm_name} ({action_type}) Loss Components on {config["env_id"]}')

    # Color scheme for different loss types - avoid blue/red used for scores
    colors = {
        "policy_loss": "orange",      # Keep orange for policy loss
        "value_loss": "purple",       # Keep purple for value loss
        "entropy_loss": "green",      # Keep green for entropy loss
        "total_loss": "darkred",      # Change from red to darkred to differentiate from continuous scores
        "actor_loss": "goldenrod",    # Change from blue to goldenrod
        "critic_loss": "darkviolet",  # Keep darkviolet for critic loss
    }

    # For loss data, x-axis represents update steps (1, 2, 3, ...)
    first_loss = list(losses_dict.values())[0]
    x_values = range(1, len(first_loss) + 1)
    x_label = "Update Step"

    # Plot all loss components on the same axes
    for loss_name, loss_values in losses_dict.items():
        color = colors.get(loss_name, "black")
        label = loss_name.replace("_", " ").title()

        # Ensure x_values and loss_values have the same length
        min_len = min(len(x_values), len(loss_values))
        x_plot = x_values[:min_len]
        y_plot = loss_values[:min_len]

        # Plot raw loss values with transparency
        ax.plot(x_plot, y_plot, label=f'{label} (Raw)', color=color, alpha=0.3)

        # Add moving average if enough data
        if len(y_plot) >= config["window_length"]:
            smoothed, _ = get_moving_average(y_plot, window=config["window_length"])
            # Calculate corresponding x values for smoothed data
            smoothed_x = x_plot[config["window_length"]-1:][:len(smoothed)]
            ax.plot(
                smoothed_x,
                smoothed,
                color=color,
                linewidth=2,
                label=f'{label} ({config["window_length"]}pt avg)',
            )

    ax.set_ylabel("Loss Value")
    ax.set_xlabel(x_label)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_results(
    scores, losses, config, action_type, algorithm_name="Algorithm"
):
    """
    Combined plotting function for backward compatibility.

    Args:
        losses: Can be either a list (legacy) or dict (new format)
    """
    # Plot scores (still uses episode numbers)
    plot_training_scores(scores, config, action_type, algorithm_name)

    # Handle both legacy list format and new dict format for losses
    if isinstance(losses, dict):
        plot_training_losses(losses, config, action_type, algorithm_name)
    else:
        # Legacy format - assume it's policy loss only
        losses_dict = {
            "policy_loss": losses,
            "total_loss": losses,  # For REINFORCE, total loss == policy loss
        }
        plot_training_losses(losses_dict, config, action_type, algorithm_name)


def plot_variance_analysis(
    agent, scores, action_type, config, algorithm_name="Algorithm"
):
    """
    Visualize variance and training stability metrics.
    Generic function that works with any algorithm that tracks gradient norms and episode scores.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"{algorithm_name} Training Analysis ({action_type} Actions)", fontsize=16
    )

    episodes = range(1, len(scores) + 1)

    # 1. Episode Scores Distribution (use episode_scores, not episode_returns)
    episode_data = getattr(agent, 'episode_scores', scores)  # Fallback to scores if no episode_scores
    ax1.hist(
        episode_data, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax1.axvline(
        np.mean(episode_data),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(episode_data):.1f}",
    )
    ax1.axvline(
        np.median(episode_data),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(episode_data):.1f}",
    )
    ax1.set_xlabel("Episode Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Episode Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gradient Norm Over Time (use update steps for gradient data)
    if len(agent.gradient_norms) > 0:
        # For gradient data, x-axis represents update steps
        x_values = range(1, len(agent.gradient_norms) + 1)
        x_label = "Update Step"
        
        ax2.plot(x_values, agent.gradient_norms, alpha=0.6, color="purple")

        # Use config window size for smoothing
        if len(agent.gradient_norms) >= config["window_length"]:
            smoothed, _ = get_moving_average(agent.gradient_norms, window=config["window_length"])
            smoothed_x = x_values[config["window_length"]-1:][:len(smoothed)]
            ax2.plot(
                smoothed_x,
                smoothed,
                color="darkviolet",
                linewidth=2,
                label=f'Smoothed ({config["window_length"]}pt)',
            )
    else:
        x_label = "Update Step"
        
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Magnitude Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Score Variance Over Time (updated terminology)
    variance_attr = getattr(agent, 'score_variance_history', getattr(agent, 'return_variance_history', []))
    variance_label = "Score Variance" if hasattr(agent, 'score_variance_history') else "Variance"
    
    if len(variance_attr) > 0:
        # Raw variance data starts at episode window_length
        variance_start_episode = config["window_length"]
        variance_episodes = range(variance_start_episode, variance_start_episode + len(variance_attr))
        
        ax3.plot(
            variance_episodes, variance_attr, color="green", alpha=0.7, label=f"Raw {variance_label}"
        )

        # Smoothed variance with correct offset calculation
        if len(variance_attr) >= config["window_length"]:
            smoothed, smoothing_offset = get_moving_average(
                variance_attr, window=config["window_length"]
            )
            smoothed_start_episode = variance_start_episode + smoothing_offset
            smoothed_episodes = range(smoothed_start_episode, smoothed_start_episode + len(smoothed))
            ax3.plot(
                smoothed_episodes,
                smoothed,
                color="darkgreen",
                linewidth=2,
                label=f'Smoothed ({config["window_length"]}ep)',
            )
    
    ax3.set_xlabel("Episode")
    ax3.set_ylabel(f"{variance_label} (last {config['window_length']} episodes)")
    ax3.set_title(f"Rolling {variance_label}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Gradient vs Score Correlation (updated terminology)
    if len(agent.gradient_norms) > 0 and len(episode_data) > 0:
        min_len = min(len(agent.gradient_norms), len(episode_data))
        grad_subset = agent.gradient_norms[:min_len]
        score_subset = episode_data[:min_len]

        ax4.scatter(score_subset, grad_subset, alpha=0.6, color="coral")

        # Calculate correlation
        if len(grad_subset) > 1:
            correlation = np.corrcoef(score_subset, grad_subset)[0, 1]
            ax4.set_title(
                f"Gradient Norm vs Episode Score\nCorrelation: {correlation:.3f}"
            )
        else:
            ax4.set_title("Gradient Norm vs Episode Score\n(Insufficient data)")

    ax4.set_xlabel("Episode Score")
    ax4.set_ylabel("Gradient Norm")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print basic statistics with corrected terminology
    stats = agent.get_variance_stats()
    print(f"\n--- {algorithm_name} Training Statistics ({action_type}) ---")
    
    # Print both score and return stats for MC methods, only scores for TD/PPO
    if hasattr(agent, 'episode_returns') and len(getattr(agent, 'episode_returns', [])) > 0:
        # MC methods: show both scores and returns
        print(f"Episode Scores: μ={stats['score_mean']:.2f}, σ={stats['score_std']:.2f}")
        print(f"Episode Returns (G_0): μ={stats['return_mean']:.2f}, σ={stats['return_std']:.2f}")
        print(f"Recent Score Variance: {stats.get('recent_score_variance', 0.0):.2f}")
        print(f"Recent Return Variance: {stats.get('recent_return_variance', 0.0):.2f}")
    else:
        # TD/PPO methods: only scores available
        print(f"Episode Scores: μ={stats.get('score_mean', stats.get('return_mean', 0.0)):.2f}, σ={stats.get('score_std', stats.get('return_std', 0.0)):.2f}")
        print(f"Recent Score Variance: {stats.get('recent_score_variance', stats.get('recent_return_variance', 0.0)):.2f}")
    
    print(
        f"Gradient Norms: μ={stats['gradient_norm_mean']:.4f}, σ={stats['gradient_norm_std']:.4f}"
    )
    if hasattr(agent, 'update_step'):
        print(f"Total Update Steps: {agent.update_step}")


def plot_ppo_training_losses(losses_dict, config, action_type, algorithm_name="PPO"):
    """
    Plot PPO training losses with rollout-based x-axis.
    
    Args:
        losses_dict: Dictionary with loss components
        config: Configuration dictionary
        action_type: "Discrete" or "Continuous"
        algorithm_name: Name of the algorithm
    """
    loss_components = list(losses_dict.keys())

    if len(loss_components) == 0:
        print("No loss data to plot")
        return

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    fig.suptitle(f'{algorithm_name} ({action_type}) Loss Components on {config["env_id"]}')

    # Color scheme for different loss types
    colors = {
        "actor_loss": "goldenrod",
        "critic_loss": "darkviolet",
        "entropy_loss": "green",
        "total_loss": "darkred",
    }

    # For PPO, x-axis represents rollouts (1, 2, 3, ...)
    first_loss = list(losses_dict.values())[0]
    x_values = range(1, len(first_loss) + 1)
    x_label = "Rollout"

    # Plot all loss components on the same axes
    for loss_name, loss_values in losses_dict.items():
        color = colors.get(loss_name, "black")
        label = loss_name.replace("_", " ").title()

        # Ensure x_values and loss_values have the same length
        min_len = min(len(x_values), len(loss_values))
        x_plot = x_values[:min_len]
        y_plot = loss_values[:min_len]

        # Plot raw loss values with transparency
        ax.plot(x_plot, y_plot, label=f'{label} (Raw)', color=color, alpha=0.3)

        # Add moving average if enough data
        if len(y_plot) >= config["window_length"]:
            smoothed, _ = get_moving_average(y_plot, window=config["window_length"])
            # Calculate corresponding x values for smoothed data
            smoothed_x = x_plot[config["window_length"]-1:][:len(smoothed)]
            ax.plot(
                smoothed_x,
                smoothed,
                color=color,
                linewidth=2,
                label=f'{label} ({config["window_length"]}pt avg)',
            )

    ax.set_ylabel("Loss Value")
    ax.set_xlabel(x_label)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_ppo_variance_analysis(agent, scores, action_type, config, algorithm_name="PPO"):
    """
    PPO-specific variance analysis with correct x-axis labels and terminology.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"{algorithm_name} Training Analysis ({action_type} Actions)", fontsize=16
    )

    episodes = range(1, len(scores) + 1)

    # 1. Episode Scores Distribution (corrected terminology)
    episode_data = getattr(agent, 'episode_scores', scores)
    ax1.hist(
        episode_data, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax1.axvline(
        np.mean(episode_data),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(episode_data):.1f}",
    )
    ax1.axvline(
        np.median(episode_data),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(episode_data):.1f}",
    )
    ax1.set_xlabel("Episode Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Episode Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gradient Norm Over Time (use rollouts for PPO)
    if len(agent.gradient_norms) > 0:
        # For PPO, x-axis represents rollouts
        x_values = range(1, len(agent.gradient_norms) + 1)
        x_label = "Rollout"
        
        ax2.plot(x_values, agent.gradient_norms, alpha=0.6, color="purple")

        # Use config window size for smoothing
        if len(agent.gradient_norms) >= config["window_length"]:
            smoothed, _ = get_moving_average(agent.gradient_norms, window=config["window_length"])
            smoothed_x = x_values[config["window_length"]-1:][:len(smoothed)]
            ax2.plot(
                smoothed_x,
                smoothed,
                color="darkviolet",
                linewidth=2,
                label=f'Smoothed ({config["window_length"]}pt)',
            )
    else:
        x_label = "Rollout"
        
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Magnitude Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Score Variance Over Time (corrected terminology)
    variance_attr = getattr(agent, 'score_variance_history', [])
    if len(variance_attr) > 0:
        # For PPO, score variance is calculated per rollout, not per episode
        variance_rollouts = range(1, len(variance_attr) + 1)
        
        ax3.plot(
            variance_rollouts, variance_attr, color="green", alpha=0.7, label="Raw Score Variance"
        )

        # Smoothed variance
        if len(variance_attr) >= config["window_length"]:
            smoothed, smoothing_offset = get_moving_average(
                variance_attr, window=config["window_length"]
            )
            smoothed_start_rollout = smoothing_offset + 1
            smoothed_rollouts = range(smoothed_start_rollout, smoothed_start_rollout + len(smoothed))
            ax3.plot(
                smoothed_rollouts,
                smoothed,
                color="darkgreen",
                linewidth=2,
                label=f'Smoothed ({config["window_length"]}pt)',
            )
    
    ax3.set_xlabel("Rollout")
    ax3.set_ylabel(f"Score Variance (last {config['window_length']} episodes)")
    ax3.set_title("Rolling Score Variance")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Gradient vs Score Correlation (corrected terminology)
    if len(agent.gradient_norms) > 0 and len(episode_data) > 0:
        min_len = min(len(agent.gradient_norms), len(episode_data))
        grad_subset = agent.gradient_norms[:min_len]
        score_subset = episode_data[:min_len]

        ax4.scatter(score_subset, grad_subset, alpha=0.6, color="coral")

        # Calculate correlation
        if len(grad_subset) > 1:
            correlation = np.corrcoef(score_subset, grad_subset)[0, 1]
            ax4.set_title(
                f"Gradient Norm vs Episode Score\nCorrelation: {correlation:.3f}"
            )
        else:
            ax4.set_title("Gradient Norm vs Episode Score\n(Insufficient data)")

    ax4.set_xlabel("Episode Score")
    ax4.set_ylabel("Gradient Norm")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics with corrected terminology
    stats = agent.get_variance_stats()
    print(f"\n--- {algorithm_name} Training Statistics ({action_type}) ---")
    print(f"Episode Scores: μ={stats.get('score_mean', stats.get('return_mean', 0.0)):.2f}, σ={stats.get('score_std', stats.get('return_std', 0.0)):.2f}")
    print(
        f"Gradient Norms: μ={stats['gradient_norm_mean']:.4f}, σ={stats['gradient_norm_std']:.4f}"
    )
    print(f"Recent Score Variance: {stats.get('recent_score_variance', stats.get('recent_return_variance', 0.0)):.2f}")
    if hasattr(agent, 'rollout_count'):
        print(f"Total Rollouts: {agent.rollout_count}")
    if hasattr(agent, 'policy_updates'):
        print(f"Total Policy Updates: {agent.policy_updates}")


def plot_ppo_training_results(scores, losses, config, action_type, algorithm_name="PPO"):
    """
    PPO-specific training results plotting.
    """
    # Plot scores (still uses episode numbers)
    plot_training_scores(scores, config, action_type, algorithm_name)

    # Plot losses with rollout-based x-axis
    plot_ppo_training_losses(losses, config, action_type, algorithm_name)
