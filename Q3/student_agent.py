import numpy as np
import torch
import os
import glob
import re

# Import your SAC agent and configuration
from sac_core import SAC_agent
from obs_normalizer import ObservationNormalizer
import config

# Helper function to find the most recent checkpoint
def find_latest_checkpoint(model_dir, env_name, model_type="actor"):
    """
    Find the most recent checkpoint based on timestep number in the filename.

    Args:
        model_dir (str): Directory containing model checkpoints
        env_name (str): Environment name prefix for model files
        model_type (str): Type of model file to search for (actor, critic, log_alpha)

    Returns:
        str: Timestep of the most recent checkpoint, or "final" if no numeric checkpoints found
    """
    # Pattern to match model files and extract timestep
    pattern = f"{env_name}_sac_{model_type}_([0-9]+).pth"

    # Find all matching files
    files = glob.glob(os.path.join(model_dir, f"{env_name}_sac_{model_type}_*.pth"))

    # Extract timesteps from filenames
    timesteps = []
    for file in files:
        match = re.search(pattern, file)
        if match:
            timesteps.append(int(match.group(1)))

    # If no numeric timesteps found, return "final"
    if not timesteps:
        return "final"

    # Return the highest timestep as a string
    return str(max(timesteps))

# Helper function to check if a model file exists
def check_model_exists(file_path):
    """
    Check if a model file exists.

    Args:
        file_path (str): Path to the model file

    Returns:
        bool: True if the file exists, False otherwise
    """
    return os.path.isfile(file_path)

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    def __init__(self):
        # --- Use parameters from config ---
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.action_low = config.ACTION_LOW
        self.action_high = config.ACTION_HIGH
        self.hidden_dim = config.NET_WIDTH
        self.device = config.DEVICE

        # --- Create action space for random fallback ---
        self.action_space = np.zeros((self.action_dim,), dtype=np.float32)

        # --- Flag to track if model was loaded successfully ---
        self.model_loaded = False

        # --- Initialize observation normalizer if enabled ---
        self.obs_normalizer = None
        if config.USE_OBS_NORMALIZATION:
            self.obs_normalizer = ObservationNormalizer(
                state_dim=self.state_dim,
                clip_range=config.OBS_NORM_CLIP,
                device=self.device
            )
            # Normalizer will be loaded along with the model

        try:
            # --- Initialize SAC Agent ---
            self.sac_agent = SAC_agent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                actor_lr=config.ACTOR_LR,
                critic_lr=config.CRITIC_LR,
                alpha_lr=config.ALPHA_LR,
                gamma=config.GAMMA,
                tau=config.TAU,
                alpha=config.ALPHA,
                target_entropy=config.TARGET_ENTROPY,
                auto_entropy_tuning=config.AUTO_ENTROPY_TUNING,
                device=self.device,
                actor_hidden_layers=config.ACTOR_HIDDEN_LAYERS,
                critic_hidden_layers=config.CRITIC_HIDDEN_LAYERS
            )

            # --- Load trained model weights ---
            load_env_name = config.LOAD_ENV_NAME
            load_timestep = config.LOAD_MODEL_TIMESTEP

            # Define paths for model files
            q3_actor_path = f"./Q3/{load_env_name}_sac_actor_{load_timestep}.pth"
            q3_critic_path = f"./Q3/{load_env_name}_sac_critic_{load_timestep}.pth"
            q3_log_alpha_path = f"./Q3/{load_env_name}_sac_log_alpha_{load_timestep}.pth"
            q3_norm_path = f"./Q3/{load_env_name}_obs_norm_{load_timestep}.npz"

            model_dir_actor_path = f"{config.MODEL_SAVE_DIR}/{load_env_name}_sac_actor_{load_timestep}.pth"
            model_dir_critic_path = f"{config.MODEL_SAVE_DIR}/{load_env_name}_sac_critic_{load_timestep}.pth"
            # We'll define the normalizer path for later use
            model_dir_norm_path = f"{config.MODEL_SAVE_DIR}/{load_env_name}_obs_norm_{load_timestep}.npz"

            # Check if model files exist in Q3 directory
            q3_actor_exists = check_model_exists(q3_actor_path)
            q3_critic_exists = check_model_exists(q3_critic_path)
            q3_log_alpha_exists = check_model_exists(q3_log_alpha_path)
            q3_norm_exists = check_model_exists(q3_norm_path)

            # Check if model files exist in model_dir
            model_dir_actor_exists = check_model_exists(model_dir_actor_path)
            model_dir_critic_exists = check_model_exists(model_dir_critic_path)
            # We don't need to check log_alpha for loading, but we'll check for normalizer
            model_dir_norm_exists = check_model_exists(model_dir_norm_path)

            # Try to load model from Q3 directory first
            if q3_actor_exists and q3_critic_exists and q3_log_alpha_exists:
                try:
                    # Temporarily modify the model_dir in the agent to load from Q3 directory
                    original_model_dir = config.MODEL_SAVE_DIR
                    config.MODEL_SAVE_DIR = "./Q3"

                    # Load the model
                    self.sac_agent.load(load_env_name, load_timestep)
                    print(f"Successfully loaded model from: ./Q3/{load_env_name}_sac_*_{load_timestep}.pth")
                    self.model_loaded = True

                    # Restore the original model_dir
                    config.MODEL_SAVE_DIR = original_model_dir

                    # Also try to load the matching normalizer if it exists
                    if self.obs_normalizer is not None and q3_norm_exists:
                        try:
                            self.obs_normalizer.load(q3_norm_path)
                            print(f"Loaded matching observation normalizer from {q3_norm_path}")
                        except Exception as norm_e:
                            print(f"Note: Could not load normalizer from Q3 directory: {norm_e}")
                except Exception as e:
                    print(f"Could not load model from Q3 directory: {e}")
                    # Restore the original model_dir if there was an error
                    if 'original_model_dir' in locals():
                        config.MODEL_SAVE_DIR = original_model_dir

            # If model wasn't loaded from Q3 directory, try to load from model_dir
            if not self.model_loaded and model_dir_actor_exists and model_dir_critic_exists:
                try:
                    # Load the model from model_dir
                    self.sac_agent.load(load_env_name, load_timestep)
                    print(f"Successfully loaded model from: {config.MODEL_SAVE_DIR}/{load_env_name}_sac_*_{load_timestep}.pth")
                    self.model_loaded = True

                    # Also try to load the matching normalizer if it exists
                    if self.obs_normalizer is not None and model_dir_norm_exists:
                        try:
                            self.obs_normalizer.load(model_dir_norm_path)
                            print(f"Loaded matching observation normalizer from {model_dir_norm_path}")
                        except Exception as norm_e:
                            print(f"Note: Could not load normalizer from model_dir: {norm_e}")
                except Exception as e:
                    print(f"Could not load model from model_dir: {e}")

            # If model still wasn't loaded, try to find the latest checkpoint
            if not self.model_loaded:
                print("Attempting to find the most recent checkpoint...")

                # First check Q3 directory for any checkpoints
                if os.path.exists("./Q3"):
                    latest_timestep = find_latest_checkpoint("./Q3", load_env_name)
                    if latest_timestep != "final":
                        try:
                            # Temporarily modify the model_dir in the agent to load from Q3 directory
                            original_model_dir = config.MODEL_SAVE_DIR
                            config.MODEL_SAVE_DIR = "./Q3"

                            # Load the model
                            self.sac_agent.load(load_env_name, latest_timestep)
                            print(f"Successfully loaded model from: ./Q3/{load_env_name}_sac_*_{latest_timestep}.pth")
                            self.model_loaded = True

                            # Restore the original model_dir
                            config.MODEL_SAVE_DIR = original_model_dir

                            # Also try to load the matching normalizer
                            if self.obs_normalizer is not None:
                                try:
                                    norm_path = f"./Q3/{load_env_name}_obs_norm_{latest_timestep}.npz"
                                    if os.path.exists(norm_path):
                                        self.obs_normalizer.load(norm_path)
                                        print(f"Loaded matching observation normalizer from {norm_path}")
                                except Exception as norm_e:
                                    print(f"Note: Could not load normalizer: {norm_e}")
                        except Exception as e:
                            print(f"Could not load latest checkpoint from Q3 directory: {e}")
                            # Restore the original model_dir if there was an error
                            if 'original_model_dir' in locals():
                                config.MODEL_SAVE_DIR = original_model_dir

                # If still not loaded, check model_dir
                if not self.model_loaded:
                    latest_timestep = find_latest_checkpoint(config.MODEL_SAVE_DIR, load_env_name)
                    if latest_timestep != "final" and latest_timestep != load_timestep:
                        try:
                            # Try to load the latest checkpoint
                            self.sac_agent.load(load_env_name, latest_timestep)
                            print(f"Successfully loaded model from: {config.MODEL_SAVE_DIR}/{load_env_name}_sac_*_{latest_timestep}.pth")
                            self.model_loaded = True

                            # Also try to load the matching normalizer
                            if self.obs_normalizer is not None:
                                try:
                                    norm_path = f"{config.MODEL_SAVE_DIR}/{load_env_name}_obs_norm_{latest_timestep}.npz"
                                    if os.path.exists(norm_path):
                                        self.obs_normalizer.load(norm_path)
                                        print(f"Loaded matching observation normalizer from {norm_path}")
                                except Exception as norm_e:
                                    print(f"Note: Could not load normalizer: {norm_e}")
                        except Exception as inner_e:
                            print(f"Could not load latest checkpoint from model_dir: {inner_e}")
                            print("No valid model checkpoint found.")
                    else:
                        print(f"No alternative checkpoints found for {load_env_name}.")

            # If we've loaded a model successfully, update the config to reflect the actual timestep used
            if self.model_loaded:
                config.LOAD_MODEL_TIMESTEP = latest_timestep if 'latest_timestep' in locals() else load_timestep

            # Set models to evaluation mode
            self.sac_agent.actor.eval()
            self.sac_agent.critic.q1.eval()
            self.sac_agent.critic.q2.eval()

        except Exception as e:
            print(f"Warning: An error occurred while loading the model: {e}")
            print("Falling back to random action selection.")

    def act(self, observation):
        """
        Select an action given an observation.

        Args:
            observation (np.ndarray): Raw observation from Gymnasium environment

        Returns:
            np.ndarray: Selected action
        """
        # If model wasn't loaded successfully, use random actions
        if not self.model_loaded:
            return np.random.uniform(low=self.action_low, high=self.action_high, size=(self.action_dim,))

        # Normalize observation if normalizer is enabled
        if self.obs_normalizer is not None:
            observation = self.obs_normalizer.normalize(observation)

        # Use the trained policy to select action deterministically
        action = self.sac_agent.select_action(observation, deterministic=True)
        return action

    def reset(self):
        """
        Reset the agent for a new episode.
        This method is called at the beginning of each episode.
        """
        # SAC is an off-policy algorithm and doesn't need episode-level resets
        pass
