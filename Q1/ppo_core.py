# ppo_core.py
import numpy as np
import copy
import torch
import math
from models import BetaActor, GaussianActor_musigma, GaussianActor_mu, Critic

class PPO_agent(object):
    def __init__(self, state_dim, action_dim, net_width, dvc,
                 distribution_type,  # 'Beta', 'GS_ms', 'GS_m'
                 actor_lr, critic_lr,
                 gamma, lambd, clip_rate, k_epochs,
                 entropy_coef, entropy_coef_decay,
                 l2_reg_critic, grad_clip_norm,
                 actor_optim_batch_size, critic_optim_batch_size,
                 action_low=-2.0, action_high=2.0  # For action mapping
                 ):
        # Init hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_width = net_width
        self.dvc = dvc
        self.distribution_type = distribution_type
        self.a_lr = actor_lr
        self.c_lr = critic_lr
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.l2_reg = l2_reg_critic  # Internal use l2_reg
        self.grad_clip_norm = grad_clip_norm
        self.a_optim_batch_size = actor_optim_batch_size
        self.c_optim_batch_size = critic_optim_batch_size
        self.action_low = action_low
        self.action_high = action_high

        # Choose distribution for the actor
        if self.distribution_type == 'Beta':
            self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        elif self.distribution_type == 'GS_ms':
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width, self.action_high).to(self.dvc)
        elif self.distribution_type == 'GS_m':
            self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width, self.action_high).to(self.dvc)
        else:
            print('Dist Error: Invalid distribution_type')
            raise ValueError("Invalid distribution_type specified in config.")
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        # Build Critic
        self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

    def _map_action_to_env_range(self, action_tensor):
        """ Maps action from model's output range to environment's range. """
        if self.distribution_type == 'Beta':
            # Beta output is [0, 1]
            return action_tensor * (self.action_high - self.action_low) + self.action_low
        # For Gaussian with tanh, actor model should handle scaling to action_high internally
        return action_tensor  # Assume Gaussian actor already scaled output

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
            if deterministic:
                raw_action = self.actor.deterministic_act(state_tensor)
                # If Beta, raw_action is [0,1], needs mapping
                if self.distribution_type == 'Beta':
                    env_action = self._map_action_to_env_range(raw_action)
                else:  # Gaussian, assume actor handles scaling
                    env_action = raw_action

                return env_action.cpu().numpy()[0], None, None
            else:
                # Actor's get_dist should return a distribution whose samples are in the "model's native range"
                dist = self.actor.get_dist(state_tensor)

                # Sample action in model's native range (pre-scaling/squashing)
                sampled_action_model_range = dist.sample()  # e.g., [0,1] for Beta, or raw Gaussian

                # Log prob calculation for the sampled action in model's native range
                logprob_a = dist.log_prob(sampled_action_model_range).sum(axis=-1)  # Sum over action_dim if > 1

                # Map the sampled action to the environment's action range
                if self.distribution_type == 'Beta':
                    # For Beta, map from [0,1] to [action_low, action_high]
                    env_action = self._map_action_to_env_range(sampled_action_model_range)
                elif self.distribution_type == 'GS_ms' or self.distribution_type == 'GS_m':
                    # For Gaussian, apply tanh squashing and scaling instead of simple clipping
                    # This makes it consistent with deterministic_act in the models
                    env_action = torch.tanh(sampled_action_model_range) * self.action_high

                # Return:
                # 1. action_to_env: The action to use with env.step()
                # 2. action_for_buffer: The action in model's native range that log_prob corresponds to
                # 3. log_prob_for_buffer: The log_prob of action_for_buffer
                return env_action.cpu().numpy()[0], sampled_action_model_range.cpu().numpy()[0], logprob_a.cpu().numpy()

    def train(self, trajectory_data):
        self.entropy_coef *= self.entropy_coef_decay

        s = trajectory_data['s'].to(self.dvc)
        a = trajectory_data['a'].to(self.dvc)  # This 'a' should be what logprob_a_hoder corresponds to.
        r = trajectory_data['r'].to(self.dvc)
        s_next = trajectory_data['s_next'].to(self.dvc)
        # logprob_a_old is from the policy that generated the action 'a'
        logprob_a_old = trajectory_data['logprob_a'].to(self.dvc)
        done = trajectory_data['done'].to(self.dvc)
        dw = trajectory_data['dw'].to(self.dvc)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)
            deltas = r + self.gamma * vs_ * (~dw) - vs  # dw ensures V(s_next)=0 if episode is truly done
            # GAE calculation
            adv = torch.zeros_like(r).to(self.dvc)
            advantage = 0.0
            for i in reversed(range(deltas.size(0))):
                advantage = deltas[i] + self.gamma * self.lambd * advantage * (~done[i])  # if done, next advantage is 0
                adv[i] = advantage
            td_target = adv + vs
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)  # Advantage normalization

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        # indices for mini-batch sampling
        num_samples = s.shape[0]
        indices = np.arange(num_samples)

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, self.a_optim_batch_size):  # Assuming a_optim_batch_size for actor
                batch_indices = indices[start_idx: start_idx + self.a_optim_batch_size]
                s_batch = s[batch_indices]
                a_batch = a[batch_indices]  # Action used to calculate logprob_a_old
                adv_batch = adv[batch_indices]
                logprob_a_old_batch = logprob_a_old[batch_indices]

                # Actor update
                dist_now = self.actor.get_dist(s_batch)
                # logprob_a_now should be for a_batch under current policy
                logprob_a_now = dist_now.log_prob(a_batch).sum(axis=-1, keepdim=True)
                dist_entropy = dist_now.entropy().sum(axis=-1, keepdim=True)

                # Ensure logprob_a_old_batch has the same shape as logprob_a_now for ratio calculation
                if logprob_a_old_batch.ndim > 1 and logprob_a_old_batch.shape[-1] > 1 and logprob_a_now.ndim == logprob_a_old_batch.ndim - 1:
                    logprob_a_old_batch_summed = logprob_a_old_batch.sum(1, keepdim=True)
                else:
                    logprob_a_old_batch_summed = logprob_a_old_batch

                ratio = torch.exp(logprob_a_now - logprob_a_old_batch_summed)

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv_batch
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                a_loss = a_loss.mean()

                self.actor_optimizer.zero_grad()
                a_loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()

            # Critic update (can use different batch size or same loop)
            for start_idx in range(0, num_samples, self.c_optim_batch_size):
                batch_indices = indices[start_idx: start_idx + self.c_optim_batch_size]
                s_batch = s[batch_indices]
                td_target_batch = td_target[batch_indices]

                c_loss = (self.critic(s_batch) - td_target_batch).pow(2).mean()
                if self.l2_reg > 0:
                    for param in self.critic.parameters():
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
                self.critic_optimizer.step()

    def save(self, EnvName, timestep):
        # Ensure model directory exists
        import os
        model_dir = "./model"  # Or get from config
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.actor.state_dict(), f"{model_dir}/{EnvName}_actor{timestep}.pth")
        torch.save(self.critic.state_dict(), f"{model_dir}/{EnvName}_critic{timestep}.pth")  # Corrected critic save name

    def load(self, EnvName, timestep):
        model_dir = "./model"  # Or get from config
        self.actor.load_state_dict(torch.load(f"{model_dir}/{EnvName}_actor{timestep}.pth", map_location=self.dvc))
        self.critic.load_state_dict(torch.load(f"{model_dir}/{EnvName}_critic{timestep}.pth", map_location=self.dvc))  # Corrected critic load name
