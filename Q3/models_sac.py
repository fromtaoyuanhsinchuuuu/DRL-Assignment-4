# models_sac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Constants for numerical stability - adjusted for better numerical stability
LOG_STD_MIN = -5.0  # Limit minimum to prevent extremely small standard deviations
LOG_STD_MAX = 2.0   # Maintain maximum standard deviation to prevent too much randomness
EPSILON = 1e-6      # Small constant to prevent log(0)

class SquashedGaussianActor(nn.Module):
    """
    Actor network for SAC, implementing a squashed Gaussian policy.
    Maps states to a Gaussian distribution over actions, which is then squashed by tanh.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, n_hidden_layers=2):
        """
        Initialize the actor network.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
            n_hidden_layers (int): Number of hidden layers
        """
        super(SquashedGaussianActor, self).__init__()

        # Build the network
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Apply orthogonal initialization to all linear layers
        gain = np.sqrt(2)  # ReLU gain

        # Initialize hidden layers
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=gain)
                torch.nn.init.constant_(layer.bias, 0.0)

        # Initialize mu_head with small values for better initial actions
        torch.nn.init.orthogonal_(self.mu_head.weight, gain=0.01)  # Small gain for initial actions
        torch.nn.init.constant_(self.mu_head.bias, 0.0)

        # Initialize log_std_head to output values near the middle of the allowed range
        log_std_mean = (LOG_STD_MIN + LOG_STD_MAX) / 2
        torch.nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        torch.nn.init.constant_(self.log_std_head.bias, log_std_mean)

        # Initialize statistics for logging
        self.mu_stats = {'mean': 0.0, 'min': 0.0, 'max': 0.0}
        self.log_std_stats = {'mean': 0.0, 'min': 0.0, 'max': 0.0}

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            tuple: Mean and log standard deviation of the Gaussian distribution
        """
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mu, log_std

    def sample(self, state):
        """
        Sample an action from the policy given a state.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            tuple: (action, log_prob)
                - action: Action tensor (squashed by tanh)
                - log_prob: Log probability of the action
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()

        # Store mu and log_std statistics for logging
        self.mu_stats = {
            'mean': mu.mean().item(),
            'min': mu.min().item(),
            'max': mu.max().item()
        }

        self.log_std_stats = {
            'mean': log_std.mean().item(),
            'min': log_std.min().item(),
            'max': log_std.max().item()
        }

        # Debug: Print log_std stats occasionally to verify they're being updated
        if not hasattr(self, 'debug_counter'):
            self.debug_counter = 0
        self.debug_counter += 1

        if self.debug_counter < 100 and self.debug_counter % 10 == 0:
            print(f"[ACTOR DEBUG] Counter {self.debug_counter}: log_std stats updated: mean={log_std.mean().item():.4f}, min={log_std.min().item():.4f}, max={log_std.max().item():.4f}")

        # Sample from Gaussian distribution
        normal = Normal(mu, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)   # Squash using tanh

        # Debug: Print pre-tanh and post-tanh action statistics more frequently at the beginning
        # Use a static counter to track calls to this method
        if not hasattr(self, 'sample_call_count'):
            self.sample_call_count = 0
        self.sample_call_count += 1

        # Print more frequently at the beginning, then less often
        if self.sample_call_count < 100 and self.sample_call_count % 10 == 0:
            print(f"[DEBUG] Sample call #{self.sample_call_count}")
            print(f"[DEBUG] Pre-tanh action (x_t) stats: mean={x_t.mean().item():.4f}, min={x_t.min().item():.4f}, max={x_t.max().item():.4f}")
            print(f"[DEBUG] Post-tanh action (y_t) stats: mean={y_t.mean().item():.4f}, min={y_t.min().item():.4f}, max={y_t.max().item():.4f}")
            print(f"[DEBUG] mu stats: mean={mu.mean().item():.4f}, min={mu.min().item():.4f}, max={mu.max().item():.4f}")
            print(f"[DEBUG] std stats: mean={std.mean().item():.4f}, min={std.min().item():.4f}, max={std.max().item():.4f}")
        elif torch.rand(1).item() < 0.001:  # Print with 0.1% probability after that
            print(f"[DEBUG] Sample call #{self.sample_call_count}")
            print(f"[DEBUG] Pre-tanh action (x_t) stats: mean={x_t.mean().item():.4f}, min={x_t.min().item():.4f}, max={x_t.max().item():.4f}")
            print(f"[DEBUG] Post-tanh action (y_t) stats: mean={y_t.mean().item():.4f}, min={y_t.min().item():.4f}, max={y_t.max().item():.4f}")
            print(f"[DEBUG] mu stats: mean={mu.mean().item():.4f}, min={mu.min().item():.4f}, max={mu.max().item():.4f}")
            print(f"[DEBUG] std stats: mean={std.mean().item():.4f}, min={std.min().item():.4f}, max={std.max().item():.4f}")

        # ===== COMPLETELY REWRITTEN LOG PROBABILITY CALCULATION =====
        # Print a message to confirm we're using the new calculation
        if not hasattr(self, 'new_calculation_message_printed'):
            print("USING COMPLETELY REWRITTEN LOG PROBABILITY CALCULATION")
            self.new_calculation_message_printed = True

        # Calculate std statistics for debugging
        std_mean = std.mean().item()
        std_min = std.min().item()
        std_max = std.max().item()

        # Step 1: Calculate Gaussian log probability
        gaussian_log_prob = normal.log_prob(x_t)

        # Step 2: Calculate log determinant of Jacobian for tanh transformation
        # This is log(1 - tanh(x)^2) = log(1 - y_t^2)
        log_abs_det_jacobian = torch.log(1 - y_t.pow(2) + EPSILON)

        # Step 3: Calculate final log probability
        # The correct formula is: log_prob = gaussian_log_prob - log_abs_det_jacobian
        log_prob_before_sum = gaussian_log_prob - log_abs_det_jacobian

        # Step 4: Ensure log_prob is never positive (which would be mathematically impossible)
        log_prob_before_sum = torch.clamp(log_prob_before_sum, max=0.0)

        # Debug information
        if self.sample_call_count < 100 and self.sample_call_count % 10 == 0:
            print(f"[DEBUG] gaussian_log_prob stats: mean={gaussian_log_prob.mean().item():.4f}, min={gaussian_log_prob.min().item():.4f}, max={gaussian_log_prob.max().item():.4f}")
            print(f"[DEBUG] log_abs_det_jacobian stats: mean={log_abs_det_jacobian.mean().item():.4f}, min={log_abs_det_jacobian.min().item():.4f}, max={log_abs_det_jacobian.max().item():.4f}")
            print(f"[DEBUG] log_prob_before_sum stats: mean={log_prob_before_sum.mean().item():.4f}, min={log_prob_before_sum.min().item():.4f}, max={log_prob_before_sum.max().item():.4f}")

            # Check for any numerical issues
            if torch.isnan(gaussian_log_prob).any() or torch.isinf(gaussian_log_prob).any():
                print(f"[WARNING] NaN or Inf values detected in gaussian_log_prob!")
            if torch.isnan(log_abs_det_jacobian).any() or torch.isinf(log_abs_det_jacobian).any():
                print(f"[WARNING] NaN or Inf values detected in log_abs_det_jacobian!")
            if torch.any(log_prob_before_sum > 0):
                print(f"[WARNING] {(log_prob_before_sum > 0).sum().item()} positive log_prob values detected before clamping!")

        # Additional debug information
        if self.sample_call_count < 100 and self.sample_call_count % 10 == 0:
            print(f"[DEBUG] VERIFICATION FOR LOG_PROB CALCULATION:")
            print(f"[DEBUG] LOG_STD range: [{LOG_STD_MIN}, {LOG_STD_MAX}]")
            print(f"[DEBUG] std stats: mean={std_mean:.4f}, min={std_min:.4f}, max={std_max:.4f}")
            print(f"[DEBUG] x_t stats: mean={x_t.mean().item():.4f}, min={x_t.min().item():.4f}, max={x_t.max().item():.4f}")
            print(f"[DEBUG] y_t stats: mean={y_t.mean().item():.4f}, min={y_t.min().item():.4f}, max={y_t.max().item():.4f}")
            print(f"[DEBUG] 1-y_t^2 stats: mean={(1-y_t.pow(2)).mean().item():.4f}, min={(1-y_t.pow(2)).min().item():.4f}, max={(1-y_t.pow(2)).max().item():.4f}")

            # Print shapes and sample values
            print(f"[DEBUG] gaussian_log_prob shape: {gaussian_log_prob.shape}, values: {gaussian_log_prob.detach().cpu().numpy().flatten()[:5]}...")
            print(f"[DEBUG] log_abs_det_jacobian shape: {log_abs_det_jacobian.shape}, values: {log_abs_det_jacobian.detach().cpu().numpy().flatten()[:5]}...")
            print(f"[DEBUG] log_prob_before_sum shape: {log_prob_before_sum.shape}, values: {log_prob_before_sum.detach().cpu().numpy().flatten()[:5]}...")

        # Use log_prob_before_sum as our per-dimension log probabilities
        log_prob = log_prob_before_sum

        # Sum across action dimensions
        # For a batch of states, log_prob should have shape [batch_size, action_dim]
        # After summing, it should have shape [batch_size, 1]
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Apply safe bounds on summed log probability
        # Ensure log_prob is never positive (which would be mathematically impossible)
        # and has a reasonable lower bound to prevent numerical issues
        log_prob = torch.clamp(log_prob, min=-30.0, max=0.0)

        # Final debug information
        if self.sample_call_count < 100 and self.sample_call_count % 10 == 0:
            print(f"[DEBUG] Final log_prob (after summing) stats: mean={log_prob.mean().item():.4f}, min={log_prob.min().item():.4f}, max={log_prob.max().item():.4f}")
            print(f"[DEBUG] Final log_prob shape: {log_prob.shape}, values: {log_prob.detach().cpu().numpy().flatten()[:5]}...")
            print(f"[DEBUG] Target entropy (for reference): {-self.log_std_head.out_features}")
        elif torch.rand(1).item() < 0.001:  # Print with 0.1% probability after that
            print(f"[DEBUG] Final log_prob stats: mean={log_prob.mean().item():.4f}, min={log_prob.min().item():.4f}, max={log_prob.max().item():.4f}")

        # Scale action to be in [-1, 1]
        action = y_t

        return action, log_prob

    def deterministic_act(self, state):
        """
        Get the deterministic action (mean of the distribution).

        Args:
            state (torch.Tensor): State tensor

        Returns:
            torch.Tensor: Deterministic action
        """
        mu, _ = self.forward(state)
        return torch.tanh(mu)

class QNetwork(nn.Module):
    """
    Q-Network for SAC, estimating the Q-value of state-action pairs.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, n_hidden_layers=2):
        """
        Initialize the Q-Network.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
            n_hidden_layers (int): Number of hidden layers
        """
        super(QNetwork, self).__init__()

        # First layer processes state
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)

        # Hidden layers
        layers = []
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, 1)

        # Apply orthogonal initialization to all linear layers
        gain = np.sqrt(2)  # ReLU gain

        # Initialize first layer
        torch.nn.init.orthogonal_(self.fc1.weight, gain=gain)
        torch.nn.init.constant_(self.fc1.bias, 0.0)

        # Initialize hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=gain)
                torch.nn.init.constant_(layer.bias, 0.0)

        # Initialize output layer with smaller gain
        torch.nn.init.orthogonal_(self.fc_out.weight, gain=0.01)
        torch.nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor

        Returns:
            torch.Tensor: Q-value
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.hidden_layers(x)
        q_value = self.fc_out(x)

        return q_value

class TwinQNetwork(nn.Module):
    """
    Twin Q-Network for SAC, implementing two Q-networks to mitigate overestimation bias.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, n_hidden_layers=2):
        """
        Initialize the Twin Q-Network.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
            n_hidden_layers (int): Number of hidden layers
        """
        super(TwinQNetwork, self).__init__()

        # Two Q-Networks
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim, n_hidden_layers)

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.

        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor

        Returns:
            tuple: Q-values from both networks
        """
        q1_value = self.q1(state, action)
        q2_value = self.q2(state, action)

        return q1_value, q2_value
