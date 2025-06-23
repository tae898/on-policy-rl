import matplotlib.pyplot as plt
import numpy as np


def get_moving_average(values, window=100):
    """Calculate moving average for smoothing noisy data."""
    if len(values) < window:
        return np.array(values)
    
    # Use np.convolve with 'valid' mode
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    
    # Create corresponding x-axis values (offset by window-1)
    x_offset = window - 1
    return smoothed, x_offset

def plot_training_results(scores, losses, config, action_type, algorithm_name="Algorithm"):
    """Plot training results with scores and losses."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'{algorithm_name} ({action_type}) Performance on {config["env_id"]}')

    # Plot scores
    ax1.plot(scores, label='Raw Score', alpha=0.3, color='blue' if action_type == 'Discrete' else 'red')
    
    # Handle moving average properly
    if len(scores) >= config["scores_window_size"]:
        smoothed, x_offset = get_moving_average(scores, window=config["scores_window_size"])
        smoothed_episodes = range(x_offset + 1, x_offset + 1 + len(smoothed))
        ax1.plot(smoothed_episodes, smoothed, 
                label=f'Smoothed Score ({config["scores_window_size"]}ep)', 
                color='blue' if action_type == 'Discrete' else 'red', linewidth=2)
    
    # LunarLander target score
    ax1.axhline(y=200, color='g', linestyle='--', label='Target Score (200)')
    
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot losses
    ax2.plot(losses, label='Loss', color='orange' if action_type == 'Discrete' else 'purple')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Episode')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.show()

def plot_variance_analysis(agent, scores, action_type, algorithm_name="Algorithm"):
    """
    Visualize variance and training stability metrics.
    Generic function that works with any algorithm that tracks gradient norms and episode returns.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{algorithm_name} Training Analysis ({action_type} Actions)', fontsize=16)
    
    episodes = range(1, len(scores) + 1)
    
    # 1. Episode Returns Distribution
    ax1.hist(agent.episode_returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(agent.episode_returns), color='red', linestyle='--', 
                label=f'Mean: {np.mean(agent.episode_returns):.1f}')
    ax1.axvline(np.median(agent.episode_returns), color='orange', linestyle='--', 
                label=f'Median: {np.median(agent.episode_returns):.1f}')
    ax1.set_xlabel('Episode Return')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Episode Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradient Norm Over Time
    if len(agent.gradient_norms) > 0:
        ax2.plot(episodes, agent.gradient_norms, alpha=0.6, color='purple')
        
        # Handle moving average with proper x-axis alignment
        if len(agent.gradient_norms) >= 20:
            smoothed, x_offset = get_moving_average(agent.gradient_norms, window=20)
            smoothed_episodes = range(x_offset + 1, x_offset + 1 + len(smoothed))
            ax2.plot(smoothed_episodes, smoothed, 
                     color='darkviolet', linewidth=2, label='Smoothed (20ep)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Magnitude Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Return Variance Over Time
    if len(agent.return_variance_history) > 0:
        variance_episodes = range(10, len(agent.return_variance_history) + 10)
        ax3.plot(variance_episodes, agent.return_variance_history, color='green', alpha=0.7)
        
        # Handle moving average for variance history
        if len(agent.return_variance_history) >= 20:
            smoothed, x_offset = get_moving_average(agent.return_variance_history, window=20)
            smoothed_episodes = range(10 + x_offset, 10 + x_offset + len(smoothed))
            ax3.plot(smoothed_episodes, smoothed, 
                     color='darkgreen', linewidth=2, label='Smoothed (20ep)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Return Variance (last 10 episodes)')
    ax3.set_title('Rolling Return Variance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Gradient vs Return Correlation
    if len(agent.gradient_norms) > 0 and len(agent.episode_returns) > 0:
        min_len = min(len(agent.gradient_norms), len(agent.episode_returns))
        grad_subset = agent.gradient_norms[:min_len]
        return_subset = agent.episode_returns[:min_len]
        
        ax4.scatter(return_subset, grad_subset, alpha=0.6, color='coral')
        
        # Calculate correlation
        if len(grad_subset) > 1:
            correlation = np.corrcoef(return_subset, grad_subset)[0, 1]
            ax4.set_title(f'Gradient Norm vs Episode Return\nCorrelation: {correlation:.3f}')
        else:
            ax4.set_title('Gradient Norm vs Episode Return\n(Insufficient data)')
            
    ax4.set_xlabel('Episode Return')
    ax4.set_ylabel('Gradient Norm')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print basic statistics without algorithm-specific commentary
    stats = agent.get_variance_stats()
    print(f"\n--- {algorithm_name} Training Statistics ({action_type}) ---")
    print(f"Episode Returns: μ={stats['return_mean']:.2f}, σ={stats['return_std']:.2f}")
    print(f"Gradient Norms: μ={stats['gradient_norm_mean']:.4f}, σ={stats['gradient_norm_std']:.4f}")
    print(f"Recent Return Variance: {stats['recent_return_variance']:.2f}")
    print(f"Coefficient of Variation (Returns): {stats['return_std']/abs(stats['return_mean']):.2f}")

def plot_comparison(discrete_results, continuous_results, config, algorithm_name="Algorithm"):
    """Plot comparison between discrete and continuous action spaces."""
    discrete_scores, discrete_agent = discrete_results
    continuous_scores, continuous_agent = continuous_results

    # Performance comparison
    discrete_final_avg = np.mean(discrete_scores[-20:]) if len(discrete_scores) >= 20 else np.mean(discrete_scores)
    continuous_final_avg = np.mean(continuous_scores[-20:]) if len(continuous_scores) >= 20 else np.mean(continuous_scores)

    print(f"\nFinal Performance (last 20 episodes average):")
    print(f"Discrete:   {discrete_final_avg:.2f}")
    print(f"Continuous: {continuous_final_avg:.2f}")

    # Variance comparison
    discrete_stats = discrete_agent.get_variance_stats()
    continuous_stats = continuous_agent.get_variance_stats()

    print(f"\nVariance Comparison:")
    print(f"                    Discrete    Continuous")
    print(f"Return StdDev:      {discrete_stats['return_std']:.2f}        {continuous_stats['return_std']:.2f}")
    print(f"Gradient StdDev:    {discrete_stats['gradient_norm_std']:.4f}      {continuous_stats['gradient_norm_std']:.4f}")

    # Combined visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Score comparison with proper moving averages
    if len(discrete_scores) >= 20:
        discrete_smoothed, discrete_offset = get_moving_average(discrete_scores, window=20)
        discrete_episodes = range(discrete_offset + 1, discrete_offset + 1 + len(discrete_smoothed))
        ax1.plot(discrete_episodes, discrete_smoothed, label='Discrete', color='blue', linewidth=2)
    
    if len(continuous_scores) >= 20:
        continuous_smoothed, continuous_offset = get_moving_average(continuous_scores, window=20)
        continuous_episodes = range(continuous_offset + 1, continuous_offset + 1 + len(continuous_smoothed))
        ax1.plot(continuous_episodes, continuous_smoothed, label='Continuous', color='red', linewidth=2)
    
    # LunarLander target score
    ax1.axhline(y=200, color='g', linestyle='--', label='Target (200)', alpha=0.7)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score (20-episode moving average)')
    ax1.set_title(f'{algorithm_name} Performance Comparison on {config["env_id"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gradient norm comparison with proper moving averages
    if len(discrete_agent.gradient_norms) >= 20:
        discrete_grad_smoothed, discrete_grad_offset = get_moving_average(discrete_agent.gradient_norms, window=20)
        discrete_grad_episodes = range(discrete_grad_offset + 1, discrete_grad_offset + 1 + len(discrete_grad_smoothed))
        ax2.plot(discrete_grad_episodes, discrete_grad_smoothed, label='Discrete', color='blue', linewidth=2)
    
    if len(continuous_agent.gradient_norms) >= 20:
        continuous_grad_smoothed, continuous_grad_offset = get_moving_average(continuous_agent.gradient_norms, window=20)
        continuous_grad_episodes = range(continuous_grad_offset + 1, continuous_grad_offset + 1 + len(continuous_grad_smoothed))
        ax2.plot(continuous_grad_episodes, continuous_grad_smoothed, label='Continuous', color='red', linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Gradient Norm (20-episode moving average)')
    ax2.set_title('Gradient Stability Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.show()
