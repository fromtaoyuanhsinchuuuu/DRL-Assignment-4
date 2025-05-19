# config.py
import torch

# --- Environment parameters ---
ENV_NAME = "humanoid-walk"
STATE_DIM = 67         # Humanoid Walk observation space dimension
ACTION_DIM = 21        # Humanoid Walk action space dimension
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
MAX_EPISODE_STEPS = 1000  # Typical episode length for Humanoid Walk

# --- SAC algorithm hyperparameters ---
# Note: This implementation uses standard SAC with the entropy term in the critic target for better stability
GAMMA = 0.99          # Discount factor
TAU = 0.005           # Target network soft update rate
ALPHA = 0.5           # Initial entropy coefficient (if not using auto-tuning)
# Target entropy for auto-tuning alpha
# For high-dimensional action spaces, we should use a smaller magnitude
# A common heuristic is -0.5 * action_dim or -0.5 * log(action_dim)
# Using a less aggressive target for Humanoid to improve stability
# Original value was -14, which might be too aggressive
TARGET_ENTROPY = -10.5  # Using -0.5 * action_dim for better stability
ACTOR_LR = 3e-4       # Actor learning rate
CRITIC_LR = 3e-4      # Critic learning rate
ALPHA_LR = 3e-4       # Alpha learning rate (for auto-tuning)
REPLAY_BUFFER_SIZE = int(3e6)  # Replay buffer size
BATCH_SIZE = 1024      # Batch size for training
POLICY_UPDATE_FREQ = 2  # Policy network update frequency
TARGET_NETWORK_UPDATE_FREQ = 1  # Target network update frequency
REWARD_SCALE = 5       # Lower reward scaling factor for better stability
AUTO_ENTROPY_TUNING = True  # Whether to automatically tune entropy coefficient
GRAD_CLIP_NORM = 1.0  # Maximum norm for gradient clipping
ALPHA_MAX_CAP = 5.0   # Maximum value for alpha to prevent explosion

# --- Network structure parameters ---
NET_WIDTH = 512       # Width of hidden layers in neural networks
ACTOR_HIDDEN_LAYERS = 3  # Number of hidden layers in actor network
CRITIC_HIDDEN_LAYERS = 3  # Number of hidden layers in critic network

# --- Training process parameters ---
MAX_TRAIN_TIMESTEPS = int(3e6)  # Total training steps
EVAL_FREQ = int(1e4)           # Evaluate model every N steps
SAVE_FREQ = int(5e4)           # Save model every N steps
MODEL_SAVE_DIR = "./model/q3_sac"     # Model save path for Q3 SAC
LOG_DIR = "./logs"             # TensorBoard log path (optional)
RANDOM_STEPS = 20000           # Number of random steps to fill replay buffer initially

# --- Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model loading parameters for evaluation ---
LOAD_ENV_NAME = ENV_NAME
LOAD_MODEL_TIMESTEP = "final"  # Or a specific number like "100000"

# --- Random seed (optional, for reproducibility) ---
SEED = 42

# --- Observation normalization ---
USE_OBS_NORMALIZATION = True  # Whether to normalize observations
OBS_NORM_CLIP = 10.0  # Clip normalized observations to this range
