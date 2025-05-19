# config.py
import torch

# --- Environment parameters ---
ENV_NAME = 'cartpole-balance'
STATE_DIM = 5         # CartPole Balance observation space dimension
ACTION_DIM = 1        # CartPole Balance action space dimension
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
MAX_EPISODE_STEPS = 1000  # Typical episode length for CartPole Balance

# --- PPO algorithm hyperparameters ---
GAMMA = 0.99          # Discount factor
LAMBDA = 0.95         # GAE lambda parameter
CLIP_RATE = 0.2       # PPO clip ratio epsilon
K_EPOCHS = 10         # Number of policy updates per data collection
T_HORIZON = 2048      # Trajectory length for data collection (N steps)
ACTOR_LR = 3e-4       # Actor learning rate
CRITIC_LR = 3e-4      # Critic learning rate
ENTROPY_COEF = 0.01   # Entropy reward coefficient
ENTROPY_COEF_DECAY = 0.999  # Entropy coefficient decay rate (optional)
L2_REG_CRITIC = 1e-3  # Critic L2 regularization coefficient (optional)
GRAD_CLIP_NORM = 0.5  # Gradient clipping norm (optional)

# --- Network structure parameters ---
NET_WIDTH = 128       # Width of hidden layers in neural networks
# Actor output distribution type: 'Beta', 'GS_ms' (Gaussian mu+sigma), 'GS_m' (Gaussian mu, fixed/learned sigma)
DISTRIBUTION_TYPE = 'GS_ms'  # Gaussian with learned std is suitable for CartPole Balance

# --- Training process parameters ---
MAX_TRAIN_TIMESTEPS = int(3e5)  # Total training steps
EVAL_FREQ = int(5e3)           # Evaluate model every N steps
SAVE_FREQ = int(2e4)           # Save model every N steps
MODEL_SAVE_DIR = "./model"     # Model save path
LOG_DIR = "./logs"             # TensorBoard log path (optional)

# --- Mini-batch parameters ---
# Note: T_HORIZON should be divisible by BATCH_SIZE, or handle remainders in training loop
# BATCH_SIZE refers to mini-batch size used for each update during K_EPOCHS
ACTOR_OPTIM_BATCH_SIZE = 64
CRITIC_OPTIM_BATCH_SIZE = 64

# --- Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model loading parameters for evaluation ---
LOAD_ENV_NAME = ENV_NAME
LOAD_MODEL_TIMESTEP = "final"  # Or a specific number like "100000"

# --- Random seed (optional, for reproducibility) ---
SEED = 42
