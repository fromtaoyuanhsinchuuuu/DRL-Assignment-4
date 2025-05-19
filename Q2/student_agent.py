# student_agent.py
import numpy as np
import torch
import os

# Import your PPO core class and configuration
from ppo_core import PPO_agent
import config  # Import hyperparameters

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    def __init__(self):
        # --- Use parameters from config ---
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.action_low = config.ACTION_LOW
        self.action_high = config.ACTION_HIGH
        self.net_width = config.NET_WIDTH
        self.dvc = config.DEVICE  # Use device from config
        self.distribution_type = config.DISTRIBUTION_TYPE

        # --- Create action space for random fallback ---
        self.action_space = np.zeros((self.action_dim,), dtype=np.float32)
        # For CartPole Balance, action space is [-1.0, 1.0]

        # --- Flag to track if model was loaded successfully ---
        self.model_loaded = False

        try:
            # --- Initialize PPO Agent (only for loading model structure) ---
            # Training hyperparameter values don't matter, as we only load weights, but structure needs to match
            # Pass parameters necessary for building networks and select_action
            self.ppo_agent = PPO_agent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                net_width=self.net_width,
                dvc=self.dvc,
                distribution_type=self.distribution_type,
                # These training parameters won't be used during evaluation, but PPO_agent's __init__ may need them
                actor_lr=config.ACTOR_LR,  # Placeholder
                critic_lr=config.CRITIC_LR,  # Placeholder
                gamma=config.GAMMA,
                lambd=config.LAMBDA,
                clip_rate=config.CLIP_RATE,
                k_epochs=config.K_EPOCHS,
                entropy_coef=config.ENTROPY_COEF,
                entropy_coef_decay=config.ENTROPY_COEF_DECAY,
                l2_reg_critic=config.L2_REG_CRITIC,
                grad_clip_norm=config.GRAD_CLIP_NORM,
                actor_optim_batch_size=config.ACTOR_OPTIM_BATCH_SIZE,
                critic_optim_batch_size=config.CRITIC_OPTIM_BATCH_SIZE,
                action_low=self.action_low,
                action_high=self.action_high
            )

            # --- Load trained model weights ---
            load_env_name = config.LOAD_ENV_NAME
            load_timestep = config.LOAD_MODEL_TIMESTEP

            # Ensure PPO_agent's load method uses correct path and map_location
            self.ppo_agent.load(EnvName=load_env_name, timestep=load_timestep)
            print(f"Successfully loaded model from: {config.MODEL_SAVE_DIR}/{load_env_name}_*{load_timestep}.pth")
            self.model_loaded = True

            # Set models to evaluation mode
            self.ppo_agent.actor.eval()
            if hasattr(self.ppo_agent, 'critic'):
                self.ppo_agent.critic.eval()
        except Exception as e:
            print(f"Warning: An error occurred while loading the model: {e}")
            print("Falling back to random action selection.")

    def act(self, observation):
        # If model wasn't loaded successfully, use random actions
        if not self.model_loaded:
            return np.random.uniform(low=self.action_low, high=self.action_high, size=(self.action_dim,))

        # If model was loaded, use the trained policy
        # deterministic=True means using deterministic policy during evaluation
        # select_action should return action already mapped to environment action space
        action_env, _, _ = self.ppo_agent.select_action(observation, deterministic=True)
        return action_env  # Should be numpy array, e.g., np.array([value]) for CartPole
