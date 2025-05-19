# sac_core.py
# This implementation now uses standard SAC (Soft Actor-Critic) algorithm
# as described in the paper "Soft Actor-Critic: Off-Policy Maximum Entropy
# Deep Reinforcement Learning with a Stochastic Actor" by Haarnoja et al.
#
# The critic target includes the entropy term for better stability:
# Standard SAC:  Q_target = r + γ * (min(Q1', Q2') - α * log π(a'|s'))
#
# The entropy term is still kept in the actor loss function:
# Actor loss = E[α * log π(a|s) - Q(s,a)]
#
# This modification can help stabilize training and prevent abnormally large alpha values,
# which is particularly beneficial for high-dimensional action spaces like Humanoid (21D).

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from models_sac import SquashedGaussianActor, TwinQNetwork
import config

class SAC_agent:
    """
    Soft Actor-Critic (SAC) agent implementation.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        gamma,
        tau,
        alpha,
        target_entropy,
        auto_entropy_tuning=True,
        device=torch.device("cpu"),
        actor_hidden_layers=2,
        critic_hidden_layers=2
    ):
        """
        Initialize the SAC agent.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Dimension of hidden layers
            actor_lr (float): Learning rate for actor
            critic_lr (float): Learning rate for critic
            alpha_lr (float): Learning rate for alpha (entropy coefficient)
            gamma (float): Discount factor
            tau (float): Target network update rate
            alpha (float): Initial entropy coefficient
            target_entropy (float): Target entropy for auto-tuning alpha
            auto_entropy_tuning (bool): Whether to automatically tune alpha
            device (torch.device): Device to run on
            actor_hidden_layers (int): Number of hidden layers in actor
            critic_hidden_layers (int): Number of hidden layers in critic
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.auto_entropy_tuning = auto_entropy_tuning
        self.policy_update_freq = config.POLICY_UPDATE_FREQ
        self.grad_clip_norm = config.GRAD_CLIP_NORM  # Add gradient clipping parameter

        # Initialize actor network
        self.actor = SquashedGaussianActor(
            state_dim, action_dim, hidden_dim, actor_hidden_layers
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Initialize critic networks
        self.critic = TwinQNetwork(
            state_dim, action_dim, hidden_dim, critic_hidden_layers
        ).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize target critic networks
        self.critic_target = TwinQNetwork(
            state_dim, action_dim, hidden_dim, critic_hidden_layers
        ).to(device)
        # Copy parameters from critic to target critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize entropy coefficient (alpha)
        self.target_entropy = target_entropy

        if auto_entropy_tuning:
            # Learnable entropy coefficient
            # Initialize log_alpha to log(alpha) to match our specified initial value
            # Use a scalar tensor to avoid shape mismatch issues
            # Initialize log_alpha with capped value if needed
            initial_alpha = min(alpha, config.ALPHA_MAX_CAP)
            self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device=device, dtype=torch.float32)
            # Use a higher learning rate for alpha to allow better adaptation
            adjusted_alpha_lr = alpha_lr   # Increased for better adaptation
            # adjusted_alpha_lr = alpha_lr * 0.1  # Increased for better adaptation
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=adjusted_alpha_lr)
            self.alpha = self.log_alpha.exp().item()
            print(f"[INFO] Initialized alpha to {self.alpha:.4f} with learning rate {adjusted_alpha_lr:.6f}")
        else:
            # Fixed entropy coefficient
            self.alpha = alpha

        # Training info
        self.train_step = 0

        # Initialize statistics for logging
        self.log_probs_stats = {'mean': 0.0, 'min': 0.0, 'max': 0.0}
        self.alpha_stats = {
            'entropy_diff': 0.0,
            'alpha_loss': 0.0,
            'log_alpha': self.log_alpha.item() if auto_entropy_tuning else 0.0,
            'target_entropy': target_entropy
        }

    def select_action(self, state, deterministic=False):
        """
        Select an action given a state.

        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to select action deterministically

        Returns:
            np.ndarray: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            if deterministic:
                action = self.actor.deterministic_act(state_tensor)
                return action.cpu().numpy().flatten()
            else:
                action, _ = self.actor.sample(state_tensor)
                return action.cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size):
        """
        Train the agent using a batch of experiences from the replay buffer.

        Args:
            replay_buffer (ReplayBuffer): Replay buffer to sample from
            batch_size (int): Batch size for training

        Returns:
            dict: Dictionary of training metrics
        """
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones, real_dones = replay_buffer.sample(batch_size)

        # Update critic (always) - use real_dones for proper termination handling
        q1_loss, q2_loss = self._update_critic(states, actions, rewards, next_states, real_dones)

        # Initialize actor and alpha loss values
        actor_loss = torch.tensor(0.0, device=self.device)
        alpha_loss = torch.tensor(0.0, device=self.device)

        # Update actor and alpha less frequently based on policy_update_freq
        if self.train_step % self.policy_update_freq == 0:
            actor_loss, alpha_loss = self._update_actor_and_alpha(states)
            # Update target networks after actor update
            self._update_target_networks()

        # Increment training step
        self.train_step += 1

        # Return metrics with additional statistics
        metrics = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha,
            'log_probs_mean': self.log_probs_stats['mean'],
            'log_probs_min': self.log_probs_stats['min'],
            'log_probs_max': self.log_probs_stats['max'],
            'entropy_diff': self.alpha_stats['entropy_diff'],
            'log_alpha': self.alpha_stats['log_alpha'],
            'target_entropy': self.target_entropy,
            # Add mu and log_std statistics from the actor
            'mu_mean': self.actor.mu_stats['mean'],
            'mu_min': self.actor.mu_stats['min'],
            'mu_max': self.actor.mu_stats['max'],
            'log_std_mean': self.actor.log_std_stats['mean'],
            'log_std_min': self.actor.log_std_stats['min'],
            'log_std_max': self.actor.log_std_stats['max']
        }

        return metrics

    def _update_critic(self, states, actions, rewards, next_states, dones):
        """
        Update the critic networks.

        Args:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions
            rewards (torch.Tensor): Batch of rewards
            next_states (torch.Tensor): Batch of next states
            dones (torch.Tensor): Batch of done flags

        Returns:
            tuple: (q1_loss, q2_loss)
        """
        with torch.no_grad():
            # Sample actions from the actor for next states
            # We now need log_probs for standard SAC critic target calculation
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q values
            next_q1, next_q2 = self.critic_target(next_states, next_actions)

            # Standard SAC with entropy term in critic target
            # This helps with more stable learning in complex environments
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute current Q values
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Add gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip_norm)
        self.critic_optimizer.step()

        return q1_loss, q2_loss

    def _update_actor_and_alpha(self, states):
        """
        Update the actor network and entropy coefficient (alpha).

        Args:
            states (torch.Tensor): Batch of states

        Returns:
            tuple: (actor_loss, alpha_loss)
        """
        # Sample actions from the actor
        actions, log_probs = self.actor.sample(states)

        # Track log_probs statistics
        log_probs_mean = log_probs.mean().item()
        log_probs_min = log_probs.min().item()
        log_probs_max = log_probs.max().item()

        # Store statistics for logging
        self.log_probs_stats = {
            'mean': log_probs_mean,
            'min': log_probs_min,
            'max': log_probs_max
        }

        # Debug: Print mu and log_std statistics for verification
        if self.train_step < 100 and self.train_step % 10 == 0:
            print(f"[DEBUG] Step {self.train_step}: mu stats from actor: mean={self.actor.mu_stats['mean']:.4f}, min={self.actor.mu_stats['min']:.4f}, max={self.actor.mu_stats['max']:.4f}")
            print(f"[DEBUG] Step {self.train_step}: log_std stats from actor: mean={self.actor.log_std_stats['mean']:.4f}, min={self.actor.log_std_stats['min']:.4f}, max={self.actor.log_std_stats['max']:.4f}")

        # Print log_probs statistics more frequently at the beginning of training
        # to ensure they're being calculated correctly
        if self.train_step < 1000 and self.train_step % 10 == 0:
            print(f"[DEBUG] Step {self.train_step}: log_probs stats: mean={log_probs_mean:.4f}, min={log_probs_min:.4f}, max={log_probs_max:.4f}")
            print(f"[DEBUG] log_probs shape: {log_probs.shape}, values: {log_probs.detach().cpu().numpy().flatten()[:5]}...")
            print(f"[DEBUG] target_entropy={self.target_entropy:.4f}, alpha={self.alpha:.4f}")
            if self.auto_entropy_tuning:
                print(f"[DEBUG] log_alpha={self.log_alpha.item():.4f}")
            print(f"[DEBUG] log_probs + target_entropy mean: {(log_probs + self.target_entropy).mean().item():.4f}")
        # Regular logging every 100 steps after that
        elif self.train_step % 100 == 0:
            print(f"[DEBUG] Step {self.train_step}: log_probs stats: mean={log_probs_mean:.4f}, min={log_probs_min:.4f}, max={log_probs_max:.4f}")
            print(f"[DEBUG] target_entropy={self.target_entropy:.4f}, alpha={self.alpha:.4f}")
            if self.auto_entropy_tuning:
                print(f"[DEBUG] log_alpha={self.log_alpha.item():.4f}")
            print(f"[DEBUG] log_probs + target_entropy mean: {(log_probs + self.target_entropy).mean().item():.4f}")

        # Compute Q values for these actions
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)

        # Compute actor loss
        actor_loss = (self.alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Add gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm)
        self.actor_optimizer.step()

        # Calculate entropy difference for logging (even if not auto-tuning)
        # Calculate entropy difference as (log_probs_mean - target_entropy)
        # Positive when entropy is too low -> should INCREASE alpha
        # Negative when entropy is too high -> should DECREASE alpha
        entropy_diff = log_probs.mean().detach() - self.target_entropy
        entropy_diff_mean = entropy_diff.item()

        # Update alpha (if auto-tuning)
        alpha_loss = torch.tensor(0.0, device=self.device)
        alpha_loss_value = 0.0

        if self.auto_entropy_tuning:
            # 新版 alpha_loss debug print
            print("USING NEW ALPHA LOSS FORMULA V3")
            log_probs_mean_detached = log_probs.detach().mean()
            alpha_loss_for_grad = -self.log_alpha * (log_probs_mean_detached - self.target_entropy)
            alpha_loss_value_for_log = alpha_loss_for_grad.item()

            print(f"--- ALPHA DEBUG START (Step: {self.train_step}) ---")
            print(f"log_probs_mean_detached: {log_probs_mean_detached.item():.4f}")
            print(f"self.target_entropy (from config): {self.target_entropy:.4f}")
            print(f"self.log_alpha (before optim step): {self.log_alpha.item():.4f}")
            print(f"self.alpha (before optim step): {self.alpha:.4f}")
            print(f"alpha_loss_tensor_for_grad: {alpha_loss_for_grad.item():.4f}")

            log_alpha_before_step = self.log_alpha.clone().detach()

            self.alpha_optimizer.zero_grad()
            alpha_loss_for_grad.backward()
            if self.log_alpha.grad is not None:
                print(f"self.log_alpha.grad: {self.log_alpha.grad.item():.4f}")
            else:
                print("self.log_alpha.grad is None")
            self.alpha_optimizer.step()

            # Calculate alpha after optimization step
            alpha_after_step = self.log_alpha.exp().item()

            # Apply alpha cap to prevent explosion
            if alpha_after_step > config.ALPHA_MAX_CAP:
                print(f"[WARNING] Alpha capped from {alpha_after_step:.4f} to {config.ALPHA_MAX_CAP}")
                # Set alpha to the maximum cap
                self.alpha = config.ALPHA_MAX_CAP
                # Update log_alpha to match the capped alpha value
                with torch.no_grad():
                    self.log_alpha.fill_(np.log(config.ALPHA_MAX_CAP))
                # Get the updated value after capping
                alpha_after_step = config.ALPHA_MAX_CAP
            else:
                # If alpha is within bounds, use it as is
                self.alpha = alpha_after_step

            print(f"self.log_alpha (after optim step): {self.log_alpha.item():.4f}")
            print(f"self.alpha (after optim step): {alpha_after_step:.4f}")
            print(f"--- ALPHA DEBUG END ---")
            alpha_loss = torch.tensor(alpha_loss_value_for_log, device=self.device)

        # Store alpha-related statistics for logging
        self.alpha_stats = {
            'entropy_diff': entropy_diff_mean,
            'alpha_loss': alpha_loss_value,
            'log_alpha': self.log_alpha.item() if self.auto_entropy_tuning else np.log(self.alpha),
            'target_entropy': self.target_entropy
        }

        # Debug: Print alpha loss components
        if self.train_step % 100 == 0:
            print(f"[DEBUG] (log_probs_mean - target_entropy)={entropy_diff_mean:.4f}, alpha_loss={alpha_loss_value:.4f}")
            print(f"[DEBUG] Should {'INCREASE' if entropy_diff_mean > 0 else 'DECREASE'} alpha based on entropy_diff")
            # Also print what our alpha_loss formula is doing
            if self.auto_entropy_tuning:
                print(f"[DEBUG] Alpha is being {'INCREASED' if alpha_loss < 0 else 'DECREASED'} based on alpha_loss")

        return actor_loss, alpha_loss

    def _update_target_networks(self):
        """
        Soft update of target networks.
        """
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, env_name, timestep):
        """
        Save the agent's models.

        Args:
            env_name (str): Environment name
            timestep (str): Current timestep
        """
        model_dir = config.MODEL_SAVE_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), f"{model_dir}/{env_name}_sac_actor_{timestep}.pth")
        torch.save(self.critic.state_dict(), f"{model_dir}/{env_name}_sac_critic_{timestep}.pth")

        if self.auto_entropy_tuning:
            torch.save(self.log_alpha, f"{model_dir}/{env_name}_sac_log_alpha_{timestep}.pth")

    def load(self, env_name, timestep):
        """
        Load the agent's models.

        Args:
            env_name (str): Environment name
            timestep (str): Timestep to load
        """
        model_dir = config.MODEL_SAVE_DIR
        self.actor.load_state_dict(
            torch.load(f"{model_dir}/{env_name}_sac_actor_{timestep}.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{model_dir}/{env_name}_sac_critic_{timestep}.pth", map_location=self.device)
        )

        # Load target critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.auto_entropy_tuning:
            self.log_alpha = torch.load(
                f"{model_dir}/{env_name}_sac_log_alpha_{timestep}.pth", map_location=self.device
            )
            # Apply alpha cap when loading model
            alpha_loaded = self.log_alpha.exp().item()
            if alpha_loaded > config.ALPHA_MAX_CAP:
                print(f"[WARNING] Loaded alpha capped from {alpha_loaded:.4f} to {config.ALPHA_MAX_CAP}")
                # Set alpha to the maximum cap
                self.alpha = config.ALPHA_MAX_CAP
                # Update log_alpha to match the capped alpha value
                with torch.no_grad():
                    self.log_alpha.fill_(np.log(config.ALPHA_MAX_CAP))
            else:
                self.alpha = alpha_loaded
