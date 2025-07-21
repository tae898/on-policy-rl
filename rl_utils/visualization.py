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
        "policy_loss": "goldenrod",   # same as actor loss
        "value_loss": "darkviolet",   # same as critic loss
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


def plot_vectorized_training_scores(scores, config, action_type, algorithm_name="Algorithm"):
    """Plot training scores for vectorized environments with proper episode axis."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    fig.suptitle(f'{algorithm_name} ({action_type}) Scores on {config["env_id"]} ({config["num_envs"]} parallel envs)')

    # For vectorized environments, episodes go from 1 to config["episodes"]
    vectorized_episodes = range(1, len(scores) + 1)
    
    # Plot scores
    ax.plot(
        vectorized_episodes,
        scores,
        label="Vectorized Episode Score",
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
            label=f'Smoothed Score ({config["window_length"]} vec-ep)',
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
    ax.set_xlabel("Vectorized Episode (Averaged across parallel envs)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, config["episodes"])  # Ensure x-axis shows intended episode count

    plt.tight_layout()
    plt.show()


def plot_vectorized_training_losses(losses_dict, config, action_type, algorithm_name="Algorithm"):
    """
    Plot training losses for vectorized environments with update step axis.
    """
    loss_components = list(losses_dict.keys())

    if len(loss_components) == 0:
        print("No loss data to plot")
        return

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    fig.suptitle(f'{algorithm_name} ({action_type}) Loss Components on {config["env_id"]} ({config["num_envs"]} parallel envs)')

    # Color scheme for different loss types
    colors = {
        "actor_loss": "goldenrod",
        "critic_loss": "darkviolet", 
        "total_loss": "darkred",
    }

    # For vectorized environments, x-axis represents update steps (1, 2, 3, ...)
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


def plot_vectorized_variance_analysis(agent, scores, action_type, config, algorithm_name="Algorithm"):
    """
    Visualize variance and training stability metrics for vectorized environments.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"{algorithm_name} Training Analysis ({action_type} Actions) - {config['num_envs']} Parallel Envs", fontsize=16
    )

    vectorized_episodes = range(1, len(scores) + 1)

    # 1. Episode Scores Distribution
    ax1.hist(
        scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax1.axvline(
        np.mean(scores),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(scores):.1f}",
    )
    ax1.axvline(
        np.median(scores),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(scores):.1f}",
    )
    ax1.set_xlabel("Vectorized Episode Score (Averaged across parallel envs)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Vectorized Episode Scores")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gradient Norm Over Time (use update steps)
    if len(agent.gradient_norms) > 0:
        x_values = range(1, len(agent.gradient_norms) + 1)
        x_label = "Update Step"
        
        ax2.plot(x_values, agent.gradient_norms, alpha=0.6, color="purple")

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
    ax2.set_title("Gradient Magnitude Over Updates")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Score Variance Over Time - FIXED to properly align with vectorized episodes
    variance_attr = getattr(agent, 'score_variance_history', [])
    if len(variance_attr) > 0:
        # Score variance is calculated per vectorized episode, starting from vectorized episode 1
        # Each entry in score_variance_history corresponds to one vectorized episode
        variance_episodes = range(1, len(variance_attr) + 1)
        
        ax3.plot(
            variance_episodes, variance_attr, color="green", alpha=0.7, label="Score Variance", marker='o', markersize=4
        )

        if len(variance_attr) >= config["window_length"]:
            smoothed, smoothing_offset = get_moving_average(
                variance_attr, window=config["window_length"]
            )
            smoothed_start_episode = smoothing_offset + 1
            smoothed_episodes = range(smoothed_start_episode, smoothed_start_episode + len(smoothed))
            ax3.plot(
                smoothed_episodes,
                smoothed,
                color="darkgreen",
                linewidth=2,
                label=f'Smoothed ({config["window_length"]} vec-ep)',
            )
    
    ax3.set_xlabel("Vectorized Episode")
    ax3.set_ylabel(f"Score Variance (last {config['window_length']} vectorized episodes)")
    ax3.set_title("Rolling Score Variance")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Set x-axis limits to match the actual number of vectorized episodes
    if len(variance_attr) > 0:
        ax3.set_xlim(1, len(variance_attr))
    else:
        ax3.set_xlim(1, config["episodes"])  # Fallback to config episodes

    # 4. Gradient vs Score Correlation
    if len(agent.gradient_norms) > 0 and len(scores) > 0:
        min_len = min(len(agent.gradient_norms), len(scores))
        grad_subset = agent.gradient_norms[:min_len]
        score_subset = scores[:min_len]

        ax4.scatter(score_subset, grad_subset, alpha=0.6, color="coral")

        if len(grad_subset) > 1:
            correlation = np.corrcoef(score_subset, grad_subset)[0, 1]
            ax4.set_title(
                f"Gradient Norm vs Vectorized Episode Score\nCorrelation: {correlation:.3f}"
            )
        else:
            ax4.set_title("Gradient Norm vs Vectorized Episode Score\n(Insufficient data)")

    ax4.set_xlabel("Vectorized Episode Score (Averaged across parallel envs)")
    ax4.set_ylabel("Gradient Norm")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    stats = agent.get_variance_stats()
    print(f"\n--- {algorithm_name} Training Statistics ({action_type}) ---")
    print(f"Vectorized Episode Scores: μ={stats.get('score_mean', 0.0):.2f}, σ={stats.get('score_std', 0.0):.2f}")
    print(f"Recent Score Variance: {stats.get('recent_score_variance', 0.0):.2f}")
    print(
        f"Gradient Norms: μ={stats['gradient_norm_mean']:.4f}, σ={stats['gradient_norm_std']:.4f}"
    )
    if hasattr(agent, 'update_step'):
        print(f"Total Update Steps: {agent.update_step}")
    if hasattr(agent, 'total_episodes'):
        print(f"Total Individual Episodes: {agent.total_episodes} (across {config['num_envs']} parallel envs)")

    # Add debug info for variance tracking
    print(f"Vectorized Episodes Completed: {getattr(agent, 'vectorized_episodes', 0)}")
    print(f"Score Variance History Length: {len(variance_attr)}")

def plot_vectorized_training_results(scores, losses, config, action_type, algorithm_name="Algorithm"):
    """
    Combined plotting function for vectorized environments.
    """
    plot_vectorized_training_scores(scores, config, action_type, algorithm_name)
    plot_vectorized_training_losses(losses, config, action_type, algorithm_name)


def plot_rollout_based_training_scores(
    scores, config, action_type, algorithm_name="Algorithm"
):
    """Plot training scores for rollout-based algorithms."""
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
        alpha=0.7,
    )

    ax.set_xlabel("Vectorized Episode (averaged across environments)")
    ax.set_ylabel("Episode Score")
    ax.set_title("Training Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add performance text
    final_window = min(config["window_length"], len(scores))
    if final_window > 0:
        final_performance = np.mean(scores[-final_window:])
        ax.text(
            0.02,
            0.98,
            f"Final Performance:\n{final_performance:.1f} "
            f"(last {final_window} episodes)",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def plot_rollout_based_losses(
    losses_dict, rollout_history, neural_net_history, config, action_type,
    algorithm_name="Algorithm", per_update_losses=None
):
    """Plot losses for rollout-based algorithms with rollout/NN update x-axes."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{algorithm_name} ({action_type}) Training Losses', fontsize=16)
    
    smoothing_window = config.get("window_length", 5)
    
    # Colors for different loss components
    colors = {
        "total_loss": "darkred",
        "actor_loss": "goldenrod",
        "critic_loss": "darkviolet",
        "entropy_loss": "green",
    }

    # 1. Total Loss vs Rollouts
    if "total_loss" in losses_dict and len(losses_dict["total_loss"]) > 0:
        if len(losses_dict["total_loss"]) >= smoothing_window:
            smoothed, offset = get_moving_average(
                losses_dict["total_loss"], window=smoothing_window
            )
            x_values = range(offset + 1, offset + 1 + len(smoothed))
        else:
            smoothed = losses_dict["total_loss"]
            x_values = range(1, len(smoothed) + 1)
        
        ax1.plot(
            x_values, smoothed, color=colors["total_loss"], linewidth=2,
            label="Total Loss"
        )
        ax1.set_xlabel("Rollout Number")
        ax1.set_ylabel("Total Loss")
        ax1.set_title(f"Total Loss vs Rollouts ({smoothing_window}-rollout avg)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # 2. Component Losses vs Rollouts
    for loss_name in ["actor_loss", "critic_loss"]:
        if loss_name in losses_dict and len(losses_dict[loss_name]) > 0:
            if len(losses_dict[loss_name]) >= smoothing_window:
                smoothed, offset = get_moving_average(
                    losses_dict[loss_name], window=smoothing_window
                )
                x_values = range(offset + 1, offset + 1 + len(smoothed))
            else:
                smoothed = losses_dict[loss_name]
                x_values = range(1, len(smoothed) + 1)
            
            color = colors.get(loss_name, "gray")
            label = loss_name.replace("_", " ").title()
            ax2.plot(x_values, smoothed, color=color, linewidth=2, label=label)

    ax2.set_xlabel("Rollout Number")
    ax2.set_ylabel("Loss Value")
    ax2.set_title(f"Loss Components vs Rollouts ({smoothing_window}-rollout avg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Total Loss vs Neural Network Updates (using fine-grained data if available)
    title_suffix = ""  # Initialize title suffix
    if per_update_losses and "total_loss" in per_update_losses:
        # Use the actual per-minibatch losses mapped to neural network updates
        loss_data = per_update_losses["total_loss"]
        nn_update_numbers = range(1, len(loss_data) + 1)
        
        if len(loss_data) >= smoothing_window:
            smoothed, offset = get_moving_average(loss_data, window=smoothing_window)
            smoothed_x = range(offset + 1, offset + 1 + len(smoothed))
        else:
            smoothed = loss_data
            smoothed_x = nn_update_numbers
            
        ax3.plot(
            smoothed_x, smoothed, color=colors["total_loss"],
            linewidth=2, label="Total Loss (per NN update)"
        )
        title_suffix = f"(fine-grained, {smoothing_window}-update avg)"
        
    elif (neural_net_history and "total_loss" in losses_dict and
            len(losses_dict["total_loss"]) > 0):
        # Fallback to approximation if per-update data not available
        if len(neural_net_history) > 0 and len(losses_dict["total_loss"]) > 0:
            # Create mapping from rollout number to actual NN update count
            rollout_to_nn_update = {}
            
            # Calculate how many NN updates happen per rollout on average
            total_nn_updates = len(neural_net_history)
            total_rollouts = len(losses_dict["total_loss"])
            
            if total_rollouts > 0:
                # Map each rollout to its corresponding NN update count
                for rollout_idx in range(total_rollouts):
                    # Calculate the NN update count at the end of this rollout
                    progress_ratio = (rollout_idx + 1) / total_rollouts
                    nn_update_idx = int(progress_ratio * total_nn_updates) - 1
                    if nn_update_idx >= 0 and nn_update_idx < len(neural_net_history):
                        rollout_to_nn_update[rollout_idx] = (
                            neural_net_history[nn_update_idx]
                        )
                    else:
                        rollout_to_nn_update[rollout_idx] = rollout_idx + 1  # fallback
                
                # Create x-values using actual NN update counts
                nn_x_values = [rollout_to_nn_update[i] for i in range(total_rollouts)]
                
                if len(losses_dict["total_loss"]) >= smoothing_window:
                    smoothed, offset = get_moving_average(
                        losses_dict["total_loss"], window=smoothing_window
                    )
                    nn_x_values = nn_x_values[offset:]
                else:
                    smoothed = losses_dict["total_loss"]
            
                ax3.plot(
                    nn_x_values, smoothed, color=colors["total_loss"],
                    linewidth=2, label="Total Loss (approx from rollouts)"
                )
                title_suffix = f"(approx from rollouts, {smoothing_window}-rollout avg)"
    
    ax3.set_xlabel("Neural Network Update Number")
    ax3.set_ylabel("Total Loss")
    ax3.set_title(f"Total Loss vs Neural Network Updates {title_suffix}")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Loss Components vs Neural Network Updates (if per-update data available)
    if per_update_losses:
        for loss_name in ["actor_loss", "critic_loss", "entropy_loss"]:
            if loss_name in per_update_losses and len(per_update_losses[loss_name]) > 0:
                loss_data = per_update_losses[loss_name]
                nn_update_numbers = range(1, len(loss_data) + 1)
                
                if len(loss_data) >= smoothing_window:
                    smoothed, offset = get_moving_average(
                        loss_data, window=smoothing_window
                    )
                    smoothed_x = range(offset + 1, offset + 1 + len(smoothed))
                else:
                    smoothed = loss_data
                    smoothed_x = nn_update_numbers
                
                color = colors.get(loss_name, "gray")
                label = loss_name.replace("_", " ").title()
                ax4.plot(smoothed_x, smoothed, color=color, linewidth=2, label=label)

        ax4.set_xlabel("Neural Network Update Number")
        ax4.set_ylabel("Loss Value")
        ax4.set_title(
            f"Loss Components vs Neural Network Updates ({smoothing_window}-update avg)"
        )
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    else:
        # Fallback: Raw loss values without smoothing (recent trend)
        recent_window = min(20, len(losses_dict.get("total_loss", [])))
        if recent_window > 0:
            recent_rollouts = range(
                max(1, len(losses_dict["total_loss"]) - recent_window + 1),
                len(losses_dict["total_loss"]) + 1
            )
            recent_total = losses_dict["total_loss"][-recent_window:]
            ax4.plot(
                recent_rollouts, recent_total, color=colors["total_loss"],
                linewidth=2, label="Total Loss", alpha=0.7
            )
            
            if "actor_loss" in losses_dict:
                recent_actor = losses_dict["actor_loss"][-recent_window:]
                ax4.plot(
                    recent_rollouts, recent_actor, color=colors["actor_loss"],
                    linewidth=1, label="Actor Loss", alpha=0.7
                )
            
            if "critic_loss" in losses_dict:
                recent_critic = losses_dict["critic_loss"][-recent_window:]
                ax4.plot(
                    recent_rollouts, recent_critic, color=colors["critic_loss"],
                    linewidth=1, label="Critic Loss", alpha=0.7
                )

        ax4.set_xlabel("Recent Rollout Number")
        ax4.set_ylabel("Loss Value")
        ax4.set_title(f"Recent Loss Trends (last {recent_window} rollouts)")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_rollout_based_training_results(
    scores, losses, rollout_history, neural_net_history, config, action_type,
    algorithm_name="Algorithm", per_update_losses=None
):
    """Combined plotting for rollout-based algorithms."""
    plot_rollout_based_training_scores(scores, config, action_type, algorithm_name)
    plot_rollout_based_losses(
        losses, rollout_history, neural_net_history, config, action_type,
        algorithm_name, per_update_losses
    )
