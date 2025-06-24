import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class PolicyNetwork(nn.Module):
    """
    Policy network for LunarLander vector observations.
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, observation_dim, action_space, is_continuous, network_config):
        super(PolicyNetwork, self).__init__()
        self.is_continuous = is_continuous
        self.action_space = action_space

        # Store action bounds for continuous actions
        if is_continuous:
            if hasattr(action_space, "low"):
                self.action_low = torch.tensor(action_space.low, dtype=torch.float32)
                self.action_high = torch.tensor(action_space.high, dtype=torch.float32)
            else:
                # LunarLander-v3 continuous action bounds: [-1.0, +1.0] for each dimension
                self.action_low = torch.tensor([-1.0, -1.0], dtype=torch.float32)
                self.action_high = torch.tensor([1.0, 1.0], dtype=torch.float32)

        # Build FC network for vector inputs
        fc_layers = []
        in_features = observation_dim
        for fc_dim in network_config["fc_out_features"]:
            fc_layers.append(nn.Linear(in_features, fc_dim))
            if network_config.get("use_layer_norm"):
                fc_layers.append(nn.LayerNorm(fc_dim))

            if network_config["activation"].lower() == "relu":
                fc_layers.append(nn.ReLU())
            elif network_config["activation"].lower() == "silu":
                fc_layers.append(nn.SiLU())
            elif network_config["activation"].lower() == "tanh":
                fc_layers.append(nn.Tanh())
            else:
                raise ValueError(
                    f"Unsupported activation: {network_config['activation']}"
                )

            if network_config["dropout_rate"] > 0:
                fc_layers.append(nn.Dropout(network_config["dropout_rate"]))
            in_features = fc_dim
        self.fc = nn.Sequential(*fc_layers)
        self.feature_dim = in_features

        # Build Action Heads
        if self.is_continuous:
            # Continuous: output mean for each action dimension (2D for LunarLander)
            action_dim = action_space.shape[0] if hasattr(action_space, "shape") else 2
            self.action_mean = nn.Linear(self.feature_dim, action_dim)
            # Use a learnable std dev for each action dimension
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # Discrete: output logits for each action (4 actions for LunarLander)
            self.action_logits = nn.Linear(self.feature_dim, action_space.n)

    def forward(self, x):
        x = self.fc(x)

        if self.is_continuous:
            mean = self.action_mean(x)
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
