# Rainbow DQN vs PPO Comparison

A configurable implementation comparing Rainbow DQN and PPO algorithms on the CartPole-v1 environment. Both algorithms have flaggable features to study the impact of different components.

## 🔧 Configuration

Both algorithms use a comprehensive configuration system with flaggable features:

### Rainbow DQN Features
```python
"rainbow": {
    "use_double_dqn": True,           # Double DQN
    "use_dueling": True,              # Dueling DQN  
    "use_prioritized_replay": False,  # Prioritized Experience Replay
    "use_multi_step": False,          # Multi-step returns
    "use_distributional": False,      # Distributional RL (C51)
    "use_noisy_networks": False,      # Noisy Networks
    "use_target_updates": True,       # Target Networks
    "use_gradient_clipping": True,    # Gradient Clipping
}
```

### PPO Features
```python
"ppo": {
    "use_clipped_objective": True,          # Clipped Objective
    "use_old_policy_storage": True,         # Old Policy Storage
    "use_multiple_epochs": True,            # Multiple Epochs per Batch
    "use_gae": True,                        # Generalized Advantage Estimation
    "use_entropy_regularization": True,     # Entropy Regularization
    "use_mini_batch_training": True,        # Mini-batch Training
    "use_advantage_normalization": True,    # Advantage Normalization
    "use_gradient_clipping": True,          # Gradient Clipping
}
```

## 📊 What Gets Compared

- **Performance**: Episode scores over time
- **Training Time**: Wall-clock time to complete training
- **Loss Curves**: Training loss evolution for both algorithms
- **Feature Impact**: Effect of enabling/disabling individual features
- **Parameter Count**: Model complexity comparison
- **Update Frequency**: Different learning strategies (online vs batch)

## 🧪 Experiment Design

- Both algorithms train for the same number of episodes
- Similar network architectures (64 hidden units)
- Each feature can be independently toggled
- Comprehensive visualization of results

## 📝 Configuration Examples

### Minimal Rainbow DQN
```python
# Turn off advanced features for simpler comparison
"use_prioritized_replay": False,
"use_distributional": False,
"use_noisy_networks": False,
"use_multi_step": False,
```

### Vanilla PPO
```python
# Disable PPO features
"use_clipped_objective": False,
"use_gae": False,
"use_entropy_regularization": False,
"use_mini_batch_training": False,
```

## 🛠️ Requirements

Run `pip install -r requirements.txt` to install all dependencies.
Both algorithms use similar network sizes for fair comparison

### Comprehensive Visualization
- Performance comparison over episodes
- Loss curves for both algorithms
- Q-value evolution (DQN)
- Entropy tracking (PPO)
- Score distribution analysis

### Fairness Metrics
- **Environment Interactions**: Primary comparison metric
- **Update Frequency Analysis**: Understanding different learning strategies
- **Time Efficiency**: Wall-clock training time comparison
- **Parameter Count**: Model complexity comparison

## 🧪 Experimental Design

### Fair Comparison Principles
1. **Same Environment Interactions**: Both algorithms train for identical episodes
2. **Similar Model Complexity**: Comparable parameter counts
3. **Optimized Hyperparameters**: Each algorithm uses its best-known settings
4. **Multiple Metrics**: Performance, efficiency, and stability

### Feature Ablation
Each feature can be toggled independently to study its impact:
```python
# Example: Disable specific Rainbow features
"use_prioritized_replay": False,
"use_distributional": False,
"use_noisy_networks": False,
```

## 🔬 Key Insights

### Algorithm Strengths
**Rainbow DQN:**
- Excellent for environments requiring precise value estimates
- Stable learning with proper feature selection
- Good sample efficiency when features are well-matched to environment

**PPO:**
- Fast training due to batch learning efficiency
- Robust across diverse environments
- Simple to tune and deploy

### Implementation Lessons
1. **Feature Selection Matters**: Not all Rainbow features help in simple environments
2. **Update Frequency Strategy**: Online vs batch learning have different efficiency profiles
3. **Fair Comparison**: Environment interactions matter more than network updates