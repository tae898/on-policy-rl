# Reinforcement Learning Teaching Series

A comprehensive educational series of Jupyter notebooks teaching reinforcement learning
algorithms from fundamentals to state-of-the-art. Each notebook demonstrates different
RL paradigms using the **CarRacing-v3 environment** for consistent comparison across all algorithms.

## 📚 Notebook Series

### 1. REINFORCE (On-Policy Policy Gradients)

- **Focus**: Pure policy gradients, Monte Carlo returns
- **Action Space**: Discrete (5 actions)
- **Key Concepts**: Policy gradient theorem, REINFORCE algorithm, high variance

### 2. Vanilla DQN (Off-Policy Value-Based)

- **Focus**: Q-learning, Bellman equations, experience replay
- **Action Space**: Discrete (5 actions)
- **Key Concepts**: Q-function approximation, target networks, ε-greedy exploration

### 3. Vanilla Actor-Critic (Bridge Methods)

- **Focus**: Combining value and policy methods, on-policy vs off-policy
- **Action Space**: Both discrete AND continuous
- **Key Concepts**: V(s) vs Q(s,a), Bellman expectation vs optimality equations
- **Variants**:
  - On-policy discrete (V-function critic)
  - Off-policy continuous (Q-function critic)

### 4. Off-Policy Continuous Control Evolution

- **Focus**: Algorithm debugging, systematic improvements, and theoretical breakthroughs
- **Action Space**: Continuous (3D action space)

**Part A: DDPG - The Foundation (and its problems)**

- Basic off-policy actor-critic for continuous control
- Key issues: overestimation bias, instability, poor exploration

**Part B: TD3 - Systematic Debugging**

- **Key Concepts**: Twin critics, delayed updates, target policy smoothing
- **Educational Focus**: How to identify and fix specific RL problems
- **Meta-Skills**: Algorithm debugging methodology, ablation studies

**Part C: SAC - Theoretical Revolution**

- **Key Concepts**: Maximum entropy framework, automatic exploration tuning
- **Focus**: Principled approach to exploration and robustness

**Part D: Algorithm Comparison & Selection**

- When to use DDPG vs TD3 vs SAC
- Performance benchmarks and practical considerations

### 5. PPO (Modern On-Policy)

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

### Algorithm Development Skills

- **Systematic Debugging**: How to identify and fix specific RL problems (TD3 focus)
- **Algorithm Evolution**: Understanding how research progresses incrementally
- **Performance Analysis**: Comparing algorithms fairly across environments
- **Implementation Trade-offs**: Complexity vs performance vs stability

### Practical Implementation

- **Action Space Handling**: Discrete vs continuous action implementations
- **Network Architectures**: CNN feature extraction for visual inputs
- **Training Stability**: Target networks, experience replay, clipping techniques
- **Hyperparameter Sensitivity**: Understanding what matters most for each algorithm

## 🏎️ Environment: CarRacing-v3

### Overview

A top-down racing environment where the agent controls a car to complete randomly generated tracks. This environment is ideal for RL education because it's visually intuitive, supports both discrete and continuous control, and provides immediate feedback on agent performance.

### Environment Specifications

**Observation Space**: `Box(0, 255, (96, 96, 3), uint8)`

- 96×96 RGB image of the car and race track from top-down view
- Visual indicators at bottom: speed, ABS sensors, steering position, gyroscope

**Action Spaces**:

_Discrete Mode_ (5 actions):

```python
0: do_nothing
1: steer_left
2: steer_right
3: gas
4: brake
```

_Continuous Mode_ (3D Box):

```python
0: steering ∈ [-1, 1]    # -1 = full left, +1 = full right
1: gas ∈ [0, 1]          # 0 = no gas, 1 = full throttle
2: braking ∈ [0, 1]      # 0 = no brake, 1 = full brake
```

**Reward Structure**:

- `-0.1` points per frame (time penalty)
- `+1000/N` points per track tile visited (where N = total tiles in track)
- `-100` points for going off-track (episode termination)
- Example: Complete track in 732 frames → 1000 - 0.1×732 = 926.8 points

**Episode Dynamics**:

- **Starting State**: Car at rest in center of road
- **Success Condition**: Visit 95% of track tiles (configurable)
- **Failure Condition**: Drive off-track or timeout
- **Track Generation**: Random track layout each episode

### Environment Configuration

```python
# Our standard configuration across all notebooks
env = gym.make(
    "CarRacing-v3",
    render_mode="rgb_array",      # or "human" for visualization
    continuous=False              # Set per algorithm needs
)
```

### Why CarRacing-v3 for RL Education?

- **Visual & Intuitive**: Everyone understands the driving task
- **Dual Action Support**: Perfect for comparing discrete vs continuous algorithms
- **Realistic Complexity**: High-dimensional visual input with spatial structure
- **Clear Objective**: Complete the track efficiently
- **Fast Feedback**: Immediate visual results for debugging and analysis
- **Scalable Difficulty**: Random tracks provide good generalization testing

## 📈 Algorithm Coverage Matrix

| Algorithm    | Paradigm   | Data Usage    | Action Space | Key Innovation                |
| ------------ | ---------- | ------------- | ------------ | ----------------------------- |
| REINFORCE    | On-Policy  | Fresh Only    | Discrete     | Pure policy gradients         |
| DQN          | Off-Policy | Replay Buffer | Discrete     | Deep Q-learning               |
| Actor-Critic | Both       | Both          | Both         | Value + Policy combination    |
| DDPG         | Off-Policy | Replay Buffer | Continuous   | Continuous control foundation |
| TD3          | Off-Policy | Replay Buffer | Continuous   | Systematic debugging          |
| SAC          | Off-Policy | Replay Buffer | Continuous   | Maximum entropy               |
| PPO          | On-Policy  | Fresh Only    | Both         | Stable policy updates         |

## 🛠️ Technical Implementation

### Shared Architecture Components

- **CNN Backbone**: Consistent feature extraction across algorithms
- **Preprocessing**: Frame stacking, action repeat, reward shaping
- **Visualization**: Training curves, action analysis, performance videos
- **Evaluation**: Standardized metrics and comparison framework

### Progressive Complexity

1. **Simple**: REINFORCE (basic policy network)
2. **Moderate**: DQN (Q-network + target network)
3. **Complex**: Actor-Critic (dual networks, multiple variants)
4. **Advanced**: DDPG/TD3/SAC (twin critics, target smoothing, entropy tuning)
5. **Mature**: PPO (GAE, clipping, robust implementation)

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
pip install gymnasium[box2d] torch torchvision matplotlib numpy
```

### Recommended Learning Path

1. Start with **REINFORCE** for policy gradient intuition
2. Learn **DQN** for value-based methods and off-policy learning
3. Master **Actor-Critic** to understand the core paradigm bridges ⭐
4. Study **TD3 evolution** for algorithm development methodology 🔧
5. Complete with **PPO** for modern on-policy best practices

## 📚 References

- **REINFORCE** – Williams, R. J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning."  
  [https://arxiv.org/abs/2010.11364](https://arxiv.org/abs/2010.11364)

- **DQN** – Mnih, V., et al. "Human-level control through deep reinforcement learning."  
  [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)

- **Double DQN** – Van Hasselt, H., Guez, A., & Silver, D. "Deep reinforcement learning with double Q-learning."  
  [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)

- **Dueling DQN** – Wang, Z., et al. "Dueling network architectures for deep reinforcement learning."  
  [https://arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581)

- **Distributional RL (C51)** – Bellemare, M. G., et al. "A distributional perspective on reinforcement learning."  
  [https://arxiv.org/abs/1707.06887](https://arxiv.org/abs/1707.06887)

- **DDPG** – Lillicrap, T. P., et al. "Continuous control with deep reinforcement learning."  
  [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)

- **TD3** – Fujimoto, S., Hoof, H., & Meger, D. "Addressing function approximation error in actor-critic methods."  
  [https://arxiv.org/abs/1802.09477](https://arxiv.org/abs/1802.09477)

- **SAC** – Haarnoja, T., et al. "Soft Actor-Critic Algorithms and Applications."  
  [https://arxiv.org/abs/1812.05905](https://arxiv.org/abs/1812.05905)

- **PPO** – Schulman, J., et al. "Proximal Policy Optimization Algorithms."  
  [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

- **Textbook** – Sutton, R. S., & Barto, A. G. "Reinforcement learning: An introduction."  
  [http://incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html)

---

_This teaching series emphasizes deep understanding over breadth, focusing on the
fundamental paradigms that underlie all modern RL algorithms. Each section is designed
to build your intuition, technical skills, and appreciation for the elegance and
complexity of reinforcement learning._
