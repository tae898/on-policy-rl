# On-Policy Reinforcement Learning: From Policy Gradients to PPO

[![DOI](https://zenodo.org/badge/1002873241.svg)](https://doi.org/10.5281/zenodo.15869035)

Educational Jupyter notebooks teaching **on-policy reinforcement learning** algorithms from fundamentals to state-of-the-art. Demonstrates the progressive evolution of policy gradient methods using **LunarLander-v3** for consistent comparison.

**Modern Relevance (2025)**: On-policy RL is the standard for fine-tuning Large Language Models (LLMs). PPO is used in RLHF for training ChatGPT, Claude, and Gemini due to its stability advantages.

**Key Theme**: Solving high variance in policy gradients through **baseline subtraction** → **bootstrapping** → **parallel data collection** → **trust regions**.

## 🚀 Environment: LunarLander-v3

Classic rocket trajectory optimization with dual action spaces perfect for RL education.

- **Observation**: `Box((8,), float32)` - Position, velocity, angle, angular velocity, leg contact
- **Discrete Actions**: 4 actions [do_nothing, fire_left, fire_main, fire_right]  
- **Continuous Actions**: `Box(-1, +1, (2,), float32)` - [main_engine, lateral_booster]
- **Rewards**: Distance/speed/angle penalties, +10 per leg contact, ±100 for crash/landing

**Why LunarLander?** Dual action spaces, rich 8D vector observations, realistic Box2D physics, clear success criteria, fast feedback, and educational value without visual complexity.

## 📚 Algorithm Progression

Five fundamental on-policy RL algorithms showcasing progression from pure policy gradients to advanced policy optimization:

### 01. REINFORCE (Pure Policy Gradients)

- **Focus**: Vanilla policy gradients with Monte Carlo returns
- **Target**: Raw episode returns $G_t$
- **Variance**: Extremely high | **Bias**: None | **Update**: Per episode

### 02. Actor-Critic (Monte Carlo)

- **Focus**: Value function baseline to reduce variance
- **Target**: $G_t - V(s_t)$ (advantage estimation)
- **Value Function**: $V(s) \rightarrow G_t$ (no Bellman equation)
- **Variance**: Reduced | **Bias**: None | **Update**: Per episode
- **Progression**: Simple running mean $\bar{G}$ → learned state-dependent $V(s_t)$

### 03. Actor-Critic (Temporal Difference)

- **Focus**: Bootstrapping and Bellman equation - bridge to modern RL
- **Target**: $r_t + \gamma V(s_{t+1}) - V(s_t)$ (TD error as advantage)
- **Value Function**: $V(s) \rightarrow r + \gamma V(s')$ (uses Bellman equation)
- **Variance**: Lower | **Update**: Per step

### 04. A2C (Advantage Actor-Critic)

- **Focus**: Parallel environments and synchronous updates
- **Innovation**: Multiple parallel environments for stable learning
- **Variance**: Low | **Update**: Per step and batch

### 05. PPO (Proximal Policy Optimization)

- **Focus**: Trust regions, clipped objectives, modern policy optimization
- **Innovation**: Clipped policy updates prevent destructive changes + GAE
- **Variance**: Very Low | **Update**: Multiple epochs per batch

## 🎯 The Variance-Bias Journey

### Core Problem: Policy Gradient Variance

**REINFORCE**: $\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot G_t]$

- High variance: $G_t$ varies wildly
- Unbiased: Uses true returns

### Solution Evolution

1. **Baselines** (AC-MC): $\nabla J(\theta) = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot (G_t - V(s_t))]$
   - Lower variance through baseline subtraction
   - Still unbiased

2. **Bootstrapping** (AC-TD): $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
   - Much lower variance via single-step updates
   - Introduces bias

3. **Parallel Collection** (A2C): Multiple environments → stable gradients

4. **Trust Regions** (PPO): Clipped objectives + GAE for controlled updates

## 📊 Performance Summary

| Algorithm      | Discrete Score | Continuous Score | Score CV (Stability) | Key Innovation        |
| -------------- | -------------- | ---------------- | -------------------- | --------------------- |
| REINFORCE      | -40.8 ❌       | 19.2 ⚠️          | 1.471-3.200 🔴       | Pure policy gradients |
| AC-MC (Global) | -50.9 ❌       | 88.3 🟡          | 0.748-1.008 🟡       | Simple baselines      |
| AC-MC (Value)  | 52.7 🟡        | 90.7 🟡          | 0.802-1.898 🟠       | Learned baselines     |
| AC-TD          | 163.0 🟢       | -43.6 ❌         | 2.390-2.413 🔴       | Bootstrapping         |
| A2C            | 152.1 🟢       | 162.2 🟢         | 0.578-0.611 🟡       | Parallel environments |
| PPO            | 235.7 🌟       | 231.4 🌟         | 0.178-0.251 🟢       | Trust regions + GAE   |

**Key Insights**: REINFORCE → PPO shows huge score point improvement. Stability (Score CV) improved from >3.0 to <0.3.

## 🔄 Core Concepts

### On-Policy vs Off-Policy

**On-Policy** (this series): Learn from current policy data only

- **Stability**: More stable, easier to tune
- **Sample Efficiency**: Lower (must collect fresh data)
- **Use Cases**: LLM fine-tuning (RLHF), robotics, continuous control

**Off-Policy** (future work): Learn from any policy data

- **Stability**: Less stable, harder to tune
- **Sample Efficiency**: Higher (reuses old data)
- **Use Cases**: Sample-limited environments

### Monte Carlo vs Temporal Difference

**Monte Carlo** (REINFORCE, AC-MC): Complete episode returns $G_t$

- Unbiased but high variance
- Must wait for episode completion

**Temporal Difference** (AC-TD, A2C, PPO): One-step lookahead $r_t + \gamma V(s_{t+1})$

- Lower variance but introduces bias
- Can learn from incomplete episodes

## 🛠️ Implementation & Getting Started

### Package Structure

```bash
├── 01.reinforce.ipynb            # Pure policy gradients
├── 02.actor-critic-mc.ipynb      # Actor-critic with Monte Carlo
├── 03.actor-critic-td.ipynb      # Actor-critic with Temporal Difference  
├── 04.a2c.ipynb                  # Advantage Actor-Critic (parallel envs)
├── 05.ppo.ipynb                  # Proximal Policy Optimization
├── rl_utils/                     # Shared utilities
│   ├── networks.py               # PolicyNetwork, ActorCriticNetwork
│   ├── environment.py            # Environment wrappers, video recording
│   └── visualization.py          # Plotting and analysis functions
└── videos/                       # Generated training videos
```

### Installation

```bash
# System dependencies
sudo apt install swig build-essential python3-dev

# Python packages  
pip install 'gymnasium[box2d]>=1.0' torch matplotlib numpy jupyter tqdm

# Clone repository
git clone https://github.com/tae898/on-policy-rl.git
cd on-policy-rl
```

## 🚀 Future Work: Off-Policy RL

Planning **[off-policy-rl](https://github.com/tae898/off-policy-rl)** covering DQN, Rainbow DQN, DDPG, TD3, and SAC.

**Key Difference**: Off-policy methods can reuse old experience data (higher sample efficiency) but are less stable than on-policy methods (which are preferred for LLM fine-tuning).

### Policy Types vs Learning Methods

|                | **Deterministic Policy**         | **Stochastic Policy**                 |
| -------------- | -------------------------------- | ------------------------------------- |
| **On-Policy**  | ⚠️ **Rare** — poor exploration   | ✅ **Standard**                       |
|                | ✅ Possible in theory            | e.g., **REINFORCE**, **A2C**, **PPO** |
| **Off-Policy** | ✅ **Standard**                  | ✅ **Less common**, but used          |
|                | e.g., **DQN**, **DDPG**, **TD3** | e.g., **SAC**                         |

### Areas We Won't Cover

- **World Model Learning**: Learning environment dynamics (state transition and reward functions) - niche field with limited practical success
- **Offline RL**: Learning from pre-collected datasets without environment interaction
- **Multi-Agent RL**: Multiple agents learning simultaneously - adds significant complexity
- **Hierarchical RL**: Learning at multiple temporal abstractions and skill levels
- **RL + Search**: Search methods are outside our scope

**Why skip these?** They are either niche or don't work well in practice yet.

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
