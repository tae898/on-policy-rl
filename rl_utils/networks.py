import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _get_activation_layer(activation_name):
    """Get activation layer by name."""
    if activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "silu":
        return nn.SiLU()
    elif activation_name.lower() == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")


def _build_fc_layers(in_features, layer_dims, network_config):
    """
    Build fully connected layers with consistent configuration.
    
    Args:
        in_features: Input feature dimension
        layer_dims: List of output dimensions for each layer
        network_config: Configuration dict containing activation, dropout_rate, use_layer_norm
    
    Returns:
        nn.Sequential: Sequential container of layers
        int: Output feature dimension
    """
    layers = []
    current_features = in_features
    
    for fc_dim in layer_dims:
        layers.append(nn.Linear(current_features, fc_dim))
        
        if network_config.get("use_layer_norm"):
            layers.append(nn.LayerNorm(fc_dim))
        
        layers.append(_get_activation_layer(network_config["activation"]))
        
        if network_config["dropout_rate"] > 0:
            layers.append(nn.Dropout(network_config["dropout_rate"]))
        
        current_features = fc_dim
    
    return nn.Sequential(*layers), current_features


def _setup_action_bounds(action_space, is_continuous):
    """Setup action bounds for continuous action spaces."""
    if not is_continuous:
        return None, None
    
    if hasattr(action_space, "low"):
        action_low = torch.tensor(action_space.low, dtype=torch.float32)
        action_high = torch.tensor(action_space.high, dtype=torch.float32)
    else:
        # LunarLander-v3 continuous action bounds: [-1.0, +1.0] for each dimension
        action_low = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        action_high = torch.tensor([1.0, 1.0], dtype=torch.float32)
    
    return action_low, action_high


class PolicyNetwork(nn.Module):
    """
    Policy network for LunarLander vector observations.
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, observation_dim, action_space, is_continuous, network_config):
        super(PolicyNetwork, self).__init__()
        self.is_continuous = is_continuous
        self.action_space = action_space

        # Setup action bounds for continuous actions
        self.action_low, self.action_high = _setup_action_bounds(action_space, is_continuous)

        # Build FC network for vector inputs
        self.fc, self.feature_dim = _build_fc_layers(
            observation_dim, network_config["fc_out_features"], network_config
        )

        # Build Action Heads
        if self.is_continuous:
            # Continuous: output mean for each action dimension (2D for LunarLander)
            action_dim = action_space.shape[0]
            self.action_mean = nn.Linear(self.feature_dim, action_dim)
            # Use a learnable std dev for each action dimension
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # Discrete: output logits for each action (4 actions for LunarLander)
            self.action_logits = nn.Linear(self.feature_dim, action_space.n)

        # Add loss tracking for algorithm-specific plotting
        self.policy_losses = []  # Track policy losses for REINFORCE

    def forward(self, x):
        x = self.fc(x)

        if self.is_continuous:
            mean = self.action_mean(x)
            # Handle both batched and unbatched inputs
            if mean.dim() == 1:
                # Unbatched input: mean shape is [action_dim]
                log_std = self.action_log_std.squeeze(0)  # Remove batch dimension from [1, action_dim] -> [action_dim]
            else:
                # Batched input: mean shape is [batch_size, action_dim]
                log_std = self.action_log_std.expand_as(mean)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
        else:
            logits = self.action_logits(x)
            dist = Categorical(logits=logits)

        return dist

    def clip_action(self, action):
        """
        Clip continuous actions to respect action space bounds.
        """
        if not self.is_continuous:
            return action

        # Ensure tensors are on the same device as the network
        device = next(self.parameters()).device

        # Move action bounds to the correct device if needed
        if self.action_low.device != device:
            self.action_low = self.action_low.to(device)
            self.action_high = self.action_high.to(device)

        # Handle tensor conversion and device placement
        if isinstance(action, torch.Tensor):
            action = action.to(device)
        else:
            action = torch.tensor(action, dtype=torch.float32, device=device)

        # Clip each dimension according to its specific bounds
        clipped_action = torch.clamp(action, self.action_low, self.action_high)

        # Convert back to numpy for environment
        return clipped_action.detach().cpu().numpy()

    def get_param_count(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_network_info(self):
        """Print detailed network information."""
        total_params = self.get_param_count()
        action_type = "Continuous" if self.is_continuous else "Discrete"
        print(
            f"PolicyNetwork ({action_type}) initialized with {total_params:,} trainable parameters."
        )

        if self.is_continuous:
            print(
                f"Action bounds: Low={self.action_low.numpy()}, High={self.action_high.numpy()}"
            )
        else:
            print(f"Action space: {self.action_space.n} discrete actions")


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for LunarLander vector observations.
    Shares feature extraction layers between actor (policy) and critic (value function).
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, observation_dim, action_space, is_continuous, network_config):
        super(ActorCriticNetwork, self).__init__()
        self.is_continuous = is_continuous
        self.action_space = action_space

        # Setup action bounds for continuous actions
        self.action_low, self.action_high = _setup_action_bounds(action_space, is_continuous)

        # Build shared FC network for vector inputs
        self.shared_fc, self.shared_feature_dim = _build_fc_layers(
            observation_dim, network_config["fc_out_features"], network_config
        )

        # Build Actor Branch (separate layers after shared features)
        actor_config = network_config.get("actor_features", None)
        if actor_config is not None:
            self.actor_fc, self.actor_feature_dim = _build_fc_layers(
                self.shared_feature_dim, actor_config, network_config
            )
        else:
            self.actor_fc = nn.Identity()
            self.actor_feature_dim = self.shared_feature_dim

        # Build Critic Branch (separate layers after shared features) 
        critic_config = network_config.get("critic_features", None)
        if critic_config is not None:
            self.critic_fc, self.critic_feature_dim = _build_fc_layers(
                self.shared_feature_dim, critic_config, network_config
            )
        else:
            self.critic_fc = nn.Identity()
            self.critic_feature_dim = self.shared_feature_dim

        # Build Actor Head (Policy)
        if self.is_continuous:
            # Continuous: output mean for each action dimension (2D for LunarLander)
            action_dim = action_space.shape[0]
            self.action_mean = nn.Linear(self.actor_feature_dim, action_dim)
            # Use a learnable std dev for each action dimension
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # Discrete: output logits for each action (4 actions for LunarLander)
            self.action_logits = nn.Linear(self.actor_feature_dim, action_space.n)

        # Build Critic Head (Value Function)
        self.value_head = nn.Linear(self.critic_feature_dim, 1)

        # Add loss tracking for algorithm-specific plotting
        self.actor_losses = []   # Track actor losses
        self.critic_losses = []  # Track critic losses
        self.total_losses = []   # Track total combined losses

    def forward(self, x):
        """
        Forward pass returning both policy distribution and state value.
        
        Returns:
            dist: Policy distribution (Categorical or Normal)
            value: State value estimate (scalar)
        """
        # Shared feature extraction
        shared_features = self.shared_fc(x)

        # Actor branch
        actor_features = self.actor_fc(shared_features)
        if self.is_continuous:
            mean = self.action_mean(actor_features)
            # Handle both batched and unbatched inputs
            if mean.dim() == 1:
                # Unbatched input: mean shape is [action_dim]
                log_std = self.action_log_std.squeeze(0)  # Remove batch dimension from [1, action_dim] -> [action_dim]
            else:
                # Batched input: mean shape is [batch_size, action_dim]
                log_std = self.action_log_std.expand_as(mean)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
        else:
            logits = self.action_logits(actor_features)
            dist = Categorical(logits=logits)

        # Critic branch
        critic_features = self.critic_fc(shared_features)
        value = self.value_head(critic_features).squeeze(-1)  # Remove last dimension to get scalar value

        return dist, value

    def clip_action(self, action):
        """
        Clip continuous actions to respect action space bounds.
        """
        if not self.is_continuous:
            return action

        # Ensure tensors are on the same device as the network
        device = next(self.parameters()).device

        # Move action bounds to the correct device if needed
        if self.action_low.device != device:
            self.action_low = self.action_low.to(device)
            self.action_high = self.action_high.to(device)

        # Handle tensor conversion and device placement
        if isinstance(action, torch.Tensor):
            action = action.to(device)
        else:
            action = torch.tensor(action, dtype=torch.float32, device=device)

        # Clip each dimension according to its specific bounds
        clipped_action = torch.clamp(action, self.action_low, self.action_high)

        # Convert back to numpy for environment
        return clipped_action.detach().cpu().numpy()

    def get_param_count(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_breakdown(self):
        """Get parameter count breakdown by component."""
        shared_params = sum(p.numel() for p in self.shared_fc.parameters() if p.requires_grad)
        
        actor_fc_params = 0
        if not isinstance(self.actor_fc, nn.Identity):
            actor_fc_params = sum(p.numel() for p in self.actor_fc.parameters() if p.requires_grad)
        
        critic_fc_params = 0
        if not isinstance(self.critic_fc, nn.Identity):
            critic_fc_params = sum(p.numel() for p in self.critic_fc.parameters() if p.requires_grad)
        
        if self.is_continuous:
            actor_head_params = (sum(p.numel() for p in self.action_mean.parameters() if p.requires_grad) +
                               self.action_log_std.numel())
        else:
            actor_head_params = sum(p.numel() for p in self.action_logits.parameters() if p.requires_grad)
        
        critic_head_params = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        
        total_actor_params = actor_fc_params + actor_head_params
        total_critic_params = critic_fc_params + critic_head_params
        
        return {
            'shared': shared_params,
            'actor_fc': actor_fc_params,
            'actor_head': actor_head_params,
            'actor_total': total_actor_params,
            'critic_fc': critic_fc_params,
            'critic_head': critic_head_params,
            'critic_total': total_critic_params,
            'total': shared_params + total_actor_params + total_critic_params
        }

    def print_network_info(self):
        """Print detailed network information."""
        param_breakdown = self.get_param_breakdown()
        action_type = "Continuous" if self.is_continuous else "Discrete"
        
        print(f"ActorCriticNetwork ({action_type}) initialized with {param_breakdown['total']:,} trainable parameters.")
        print(f"  Shared features:     {param_breakdown['shared']:,} parameters")
        print(f"  Actor branch:        {param_breakdown['actor_total']:,} parameters")
        print(f"    - Actor FC layers: {param_breakdown['actor_fc']:,} parameters")
        print(f"    - Actor head:      {param_breakdown['actor_head']:,} parameters")
        print(f"  Critic branch:       {param_breakdown['critic_total']:,} parameters")
        print(f"    - Critic FC layers:{param_breakdown['critic_fc']:,} parameters")
        print(f"    - Critic head:     {param_breakdown['critic_head']:,} parameters")

        if self.is_continuous:
            print(f"Action bounds: Low={self.action_low.numpy()}, High={self.action_high.numpy()}")
        else:
            print(f"Action space: {self.action_space.n} discrete actions")
