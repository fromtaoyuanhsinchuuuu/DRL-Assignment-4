# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        # Output layers will be defined in subclasses

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        return s

class BetaActor(Actor):
    def __init__(self, state_dim, action_dim, net_width):
        super(BetaActor, self).__init__(state_dim, action_dim, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, s):
        x = super(BetaActor, self).forward(s)
        # Add softplus to ensure alpha and beta are positive
        alpha = F.softplus(self.alpha_head(x)) + 1e-6  # Add small constant for stability
        beta = F.softplus(self.beta_head(x)) + 1e-6
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        return Beta(alpha, beta)

    def deterministic_act(self, s):
        # Return the mode or mean of the Beta distribution
        # Mean = alpha / (alpha + beta)
        alpha, beta = self.forward(s)
        # Clamp to avoid issues if alpha or beta are very small, though softplus helps
        act = alpha / (alpha + beta)
        return act  # Output is in [0,1]

class GaussianActor_musigma(Actor):
    def __init__(self, state_dim, action_dim, net_width, action_scale=1.0):  # action_scale = config.ACTION_HIGH
        super(GaussianActor_musigma, self).__init__(state_dim, action_dim, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.log_std_head = nn.Linear(net_width, action_dim)
        self.action_scale = action_scale  # e.g., 2.0 for Pendulum

    def forward(self, s):
        x = super(GaussianActor_musigma, self).forward(s)
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_dist(self, s):
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        # This is a Normal distribution for the action *before* tanh squashing and scaling
        # For PPO, often this simpler Normal distribution is used directly,
        # and actions are clipped by the environment or a clamp in select_action.
        # If you want squashed Gaussian like in SAC, the log_prob needs Jacobian correction.
        return Normal(mu, std)

    def deterministic_act(self, s):
        mu, _ = self.forward(s)
        # For PPO, often the raw mu is used, then scaled and clipped.
        # If using tanh for squashing:
        # action = torch.tanh(mu) * self.action_scale
        # return action
        # Simpler: return mu, and let select_action handle scaling/clipping
        # Or, if actor is responsible for scaling:
        return torch.tanh(mu) * self.action_scale  # Output is in [-action_scale, action_scale]

class GaussianActor_mu(Actor):  # Assumes fixed or globally learned std
    def __init__(self, state_dim, action_dim, net_width, action_scale=1.0):
        super(GaussianActor_mu, self).__init__(state_dim, action_dim, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        # Learnable log_std for all actions, or fixed
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # Learnable
        # self.log_std = torch.zeros(1, action_dim)  # Fixed, requires_grad=False
        self.action_scale = action_scale

    def forward(self, s):
        x = super(GaussianActor_mu, self).forward(s)
        mu = self.mu_head(x)
        return mu

    def get_dist(self, s):
        mu = self.forward(s)
        std = torch.exp(self.log_std.expand_as(mu))  # Expand to match mu's batch size
        return Normal(mu, std)

    def deterministic_act(self, s):
        mu = self.forward(s)
        # return mu  # Let select_action handle scaling/clipping
        return torch.tanh(mu) * self.action_scale

class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.fc3 = nn.Linear(net_width, 1)  # Output a single value V(s)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        val = self.fc3(s)
        return val
