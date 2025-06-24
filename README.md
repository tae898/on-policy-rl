# Reinforcement Learning Learning Series

A comprehensive educational series of Jupyter notebooks teaching reinforcement learning
algorithms from fundamentals to state-of-the-art. Each notebook demonstrates different
RL paradigms using **LunarLander-v3** for consistent comparison across all algorithms.

## 🚀 Our Learning Environment: LunarLander-v3

For this entire learning series, we use **LunarLander-v3** exclusively. This environment
is perfect for RL education because it offers both discrete and continuous action
spaces, rich vector observations, and clear success/failure conditions.

**Reference**: [Gymnasium Lunar
Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

### LunarLander-v3 🚀

Classic rocket trajectory optimization - land the lunar module safely on the landing
pad.

**Observation**: `Box((8,), float32)` - Position, velocity, angle, angular velocity, leg
ground contact **Action Spaces**:

- **Discrete**: 4 actions [do_nothing, fire_left, fire_main, fire_right]
- **Continuous**: `Box(-1, +1, (2,), float32)` - [main_engine_throttle,
  lateral_booster_throttle]

**Rewards**: Distance/speed penalties, angle penalties, +10 per leg contact, engine
costs, ±100 for crash/landing

### Why LunarLander-v3 for RL Education?

- **Dual Action Spaces**: Perfect for testing both discrete and continuous control
  algorithms
- **Vector Observations**: Rich, interpretable 8D state representation with
  physics-based features
- **Realistic Physics**: Box2D physics engine provides consistent, realistic dynamics
- **Clear Success Criteria**: Landing successfully gives +100, crashing gives -100
- **Fast Feedback**: Episodes are relatively short, enabling rapid experimentation
- **Educational Value**: Classic trajectory optimization problem that teaches
  fundamental RL concepts
- **No Visual Complexity**: Vector observations allow us to use simple MLPs

## 📚 Notebook Series

### 01. REINFORCE (On-Policy Policy Gradients)

- **Focus**: Pure policy gradients, Monte Carlo returns
- **Action Space**: Both discrete AND continuous
- **Key Concepts**: Policy gradient theorem, REINFORCE algorithm, high variance
- **Variants**:
  - Discrete REINFORCE (categorical policy)
  - Continuous REINFORCE (Gaussian policy)

### 02. Vanilla DQN (Off-Policy Value-Based)

- **Focus**: Q-learning, Bellman equations, experience replay
- **Action Space**: Discrete only
- **Key Concepts**: Q-function approximation, target networks, ε-greedy exploration

### 03. Rainbow DQN (Advanced Value-Based)

- **Focus**: Comprehensive DQN improvements, state-of-the-art value methods
- **Action Space**: Discrete only
- **Key Concepts**: Double DQN, Dueling networks, Prioritized replay, Noisy networks,
  Distributional RL, Multi-step learning

### 04. Vanilla Actor-Critic (Bridge Methods)

- **Focus**: Combining value and policy methods, on-policy vs off-policy
- **Action Space**: Both discrete AND continuous
- **Key Concepts**: $V(s)$ vs $Q(s,a)$, Bellman expectation vs optimality equations
- **Variants**:
  - On-policy discrete (V-function critic)
  - Off-policy continuous (Q-function critic)

### 05. A3C (Asynchronous Advantage Actor-Critic)

- **Focus**: Parallel learning, distributed training, stabilizing on-policy methods
- **Action Space**: Both discrete AND continuous
- **Key Concepts**: Asynchronous updates, shared parameters, advantage estimation,
  parallel experience collection
- **Key Innovation**: Multiple workers collecting experience in parallel, reducing
  correlation in training data

### 06. DDPG (Deep Deterministic Policy Gradients)

- **Focus**: Foundation of off-policy continuous control
- **Action Space**: Continuous only
- **Key Concepts**: Deterministic policy gradients, actor-critic for continuous actions,
  exploration noise
- **Key Issues**: Overestimation bias, instability, poor exploration

### 07. TD3 (Twin Delayed Deep Deterministic Policy Gradients)

- **Focus**: Systematic debugging and improvement of DDPG
- **Action Space**: Continuous only
- **Key Concepts**: Twin critics, delayed updates, target policy smoothing
- **Educational Focus**: How to identify and fix specific RL problems
- **Meta-Skills**: Algorithm debugging methodology, ablation studies

### 08. SAC (Soft Actor-Critic)

- **Focus**: Maximum entropy reinforcement learning
- **Action Space**: Continuous only
- **Key Concepts**: Maximum entropy framework, automatic exploration tuning, temperature
  parameter
- **Focus**: Principled approach to exploration and robustness

### 09. PPO (Proximal Policy Optimization)

- **Focus**: Trust region concepts, clipped objectives
- **Action Space**: Both discrete AND continuous
- **Key Concepts**: Importance sampling, GAE, clipped surrogate objective
- **Variants**:
  - Discrete PPO (categorical policy)
  - Continuous PPO (Gaussian policy)

## 🎯 Learning Objectives

By completing this series, you will understand:

### Core RL Paradigms

- **Monte Carlo Methods**: REINFORCE with high variance, unbiased estimates
- **Temporal Difference Learning**: DQN, Actor-Critic with bootstrapping
- **On-Policy vs Off-Policy**: Data usage patterns and sample efficiency tradeoffs
- **Value-Based vs Policy-Based**: When to learn Q(s,a) vs π(a|s) directly
- **Parallel Training**: A3C's asynchronous workers and distributed learning

### Algorithm Development Skills

- **Variance Analysis**: Understanding and measuring gradient variance in policy methods
- **Systematic Debugging**: How to identify and fix specific RL problems (TD3 focus)
- **Algorithm Evolution**: Understanding how research progresses incrementally
- **Performance Analysis**: Comparing algorithms fairly across environments
- **Implementation Trade-offs**: Complexity vs performance vs stability
- **Parallel Training**: Asynchronous learning and distributed experience collection

### Practical Implementation

- **Action Space Handling**: Discrete vs continuous action implementations
- **Network Architectures**: CNN feature extraction for visual inputs
- **Training Stability**: Target networks, experience replay, clipping techniques
- **Hyperparameter Sensitivity**: Understanding what matters most for each algorithm

## 📈 Algorithm Coverage Matrix

| Algorithm    | Paradigm   | Data Usage    | Action Space    | Key Innovation                 |
| ------------ | ---------- | ------------- | --------------- | ------------------------------ |
| REINFORCE    | On-Policy  | Fresh Only    | Both            | Pure policy gradients          |
| Vanilla DQN  | Off-Policy | Replay Buffer | Discrete Only   | Deep Q-learning                |
| Rainbow DQN  | Off-Policy | Replay Buffer | Discrete Only   | Comprehensive DQN improvements |
| Actor-Critic | Both       | Both          | Both            | Value + Policy combination     |
| A3C          | On-Policy  | Fresh Only    | Both            | Asynchronous parallel learning |
| DDPG         | Off-Policy | Replay Buffer | Continuous Only | Continuous control foundation  |
| TD3          | Off-Policy | Replay Buffer | Continuous Only | Systematic debugging           |
| SAC          | Off-Policy | Replay Buffer | Continuous Only | Maximum entropy                |
| PPO          | On-Policy  | Fresh Only    | Both            | Stable policy updates          |

## 🛠️ Technical Implementation

### Package Structure

```bash
rl/
├── README.md                      # This documentation
├── 01.reinforce.ipynb            # REINFORCE algorithm notebook
├── 02.dqn.ipynb                  # Vanilla DQN algorithm notebook
├── 03.rainbow-dqn.ipynb          # Rainbow DQN algorithm notebook
├── 04.actor-critic.ipynb         # Actor-Critic algorithm notebook
├── 05.a3c.ipynb                  # A3C algorithm notebook
├── 06.ddpg.ipynb                 # DDPG algorithm notebook
├── 07.td3.ipynb                  # TD3 algorithm notebook
├── 08.sac.ipynb                  # SAC algorithm notebook
├── 09.ppo.ipynb                  # PPO algorithm notebook
├── rl_utils/                      # Shared utility package
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration management
│   ├── environment.py            # Environment wrappers and preprocessing
│   ├── networks.py               # Neural network architectures
│   └── visualization.py          # Plotting and analysis functions
└── videos/                        # Generated training/test videos
```

### RL Utils Package

To keep notebooks focused on algorithm learning, we've extracted common infrastructure
into the `rl_utils` package:

**Environment utilities** (`rl_utils.environment`):

- `preprocess_state()`: Standardized state preprocessing for LunarLander
- `create_env_with_wrappers()`: Environment creation with video recording and statistics
- `test_agent()`: Standardized agent testing with visualization

**Neural networks** (`rl_utils.networks`):

- `PolicyNetwork`: Flexible policy network supporting both discrete and continuous
  actions
- Automatic parameter counting and network information printing
- Built-in action clipping for continuous control

**Visualization** (`rl_utils.visualization`):

- `plot_training_results()`: Standardized training curve plotting
- `plot_variance_analysis()`: REINFORCE-specific variance analysis
- `plot_comparison()`: Multi-algorithm comparison plotting
- `get_moving_average()`: Robust smoothing for noisy data

**Configuration** (`rl_utils.config`):

- `set_seeds()`: Reproducible random seeding

### Notebook Organization

Each algorithm notebook now follows a consistent structure:

1. **Algorithm Introduction**: Theory and mathematical foundation
2. **Setup**: Import utilities and create configuration
3. **Agent Implementation**: Algorithm-specific code only
4. **Discrete Training**: Training with discrete action space
5. **Continuous Training**: Training with continuous action space (if supported)
6. **Comparative Analysis**: Performance comparison and insights

This separation allows:

- **Focused Learning**: Notebooks contain only algorithm-specific content
- **Code Reuse**: Common utilities shared across all notebooks
- **Easy Maintenance**: Infrastructure improvements benefit all algorithms
- **Clean Comparison**: Standardized evaluation across different methods

### Shared Architecture Components

- **Consistent Preprocessing**: Standardized state normalization across algorithms
- **Unified Visualization**: Comparable plots and metrics across all methods
- **Parameter Tracking**: Automatic network size reporting and gradient monitoring
- **Video Generation**: Consistent video recording for all training runs
- **Statistical Analysis**: Standardized variance and performance analysis

## 🎓 Educational Philosophy

### Learning Approach

- **Concepts First**: Intuitive explanations before mathematical details
- **Visual Learning**: Extensive plots, diagrams, and training visualizations
- **Hands-on Practice**: Interactive hyperparameter exploration
- **Comparative Analysis**: Side-by-side algorithm performance
- **Real Understanding**: Show both successes and failure modes

### Skill Development

- **Algorithm Intuition**: Why each method works and when it fails
- **Implementation Skills**: Clean, readable, and efficient code
- **Debugging Mindset**: How to diagnose and fix training issues
- **Research Thinking**: How algorithms evolve and improve over time

## 🚀 Getting Started

### Prerequisites

- Basic knowledge of neural networks and PyTorch
- Understanding of Markov Decision Processes (MDPs)
- Familiarity with gradient descent optimization

### Installation

```bash
# System dependencies for Box2D environment
sudo apt install swig build-essential python3-dev

# Python packages
pip install 'gymnasium[box2d]>=1.0' torch torchvision matplotlib numpy jupyter tqdm

# Clone the repository
git clone <repository-url>
cd rl
```

### Running the Notebooks

1. Start Jupyter from the `rl/` directory:

```bash
cd rl/
jupyter notebook
```

## 📚 References

- [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696)
- [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [Actor-Critic Reinforcement Learning for Control with Stability Guarantees](https://arxiv.org/abs/2004.14288)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
