# train.py
import numpy as np
import torch
import os
import csv
from collections import deque
from tqdm import tqdm
import time
import argparse

import config
from dmc import make_dmc_env
from sac_core import SAC_agent
from replay_buffer import ReplayBuffer
from obs_normalizer import ObservationNormalizer

def setup_csv_logging(log_dir, filename, fieldnames):
    """
    Set up CSV logging.

    Args:
        log_dir (str): Directory to save logs
        filename (str): Name of the log file
        fieldnames (list): List of column names

    Returns:
        tuple: (csv_file, csv_writer)
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    filepath = os.path.join(log_dir, filename)
    csv_file = open(filepath, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    return csv_file, csv_writer

# Optional: from torch.utils.tensorboard import SummaryWriter

def evaluate_agent(env_name, agent, obs_normalizer=None, n_episodes=10, seed=None):
    """
    Evaluate the agent on the environment for n_episodes.

    Args:
        env_name (str): Environment name
        agent (SAC_agent): Agent to evaluate
        obs_normalizer (ObservationNormalizer): Observation normalizer
        n_episodes (int): Number of episodes to evaluate
        seed (int): Random seed

    Returns:
        tuple: (mean_reward, std_reward)
    """
    total_rewards = []

    # Set normalizer to evaluation mode if provided
    if obs_normalizer is not None:
        obs_normalizer.set_training_mode(False)
        
    for ep in range(n_episodes):
        # Create a new environment for each evaluation episode
        eval_env = make_dmc_env(env_name=env_name, seed=seed+ep if seed is not None else None,
                              flatten=True, use_pixels=False)

        # Reset the environment
        # The environment returns tuple (observation, info) in Gymnasium format
        (obs_raw, _) = eval_env.reset()

        # Normalize observation if normalizer is provided
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs_raw)
        else:
            obs = obs_raw

        episode_reward = 0
        episode_steps = 0

        while True:
            # Select deterministic action
            action = agent.select_action(obs, deterministic=True)

            # Take a step in the environment
            # The environment returns (observation, reward, terminated, truncated, info) in Gymnasium format
            (next_obs_raw, reward, terminated, truncated, _) = eval_env.step(action)

            # Normalize next observation if normalizer is provided
            if obs_normalizer is not None:
                next_obs = obs_normalizer.normalize(next_obs_raw)
            else:
                next_obs = next_obs_raw

            # Update observation and reward
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            # Check if episode is done
            done = terminated or truncated or episode_steps >= config.MAX_EPISODE_STEPS
            if done:
                break

        total_rewards.append(episode_reward)
        eval_env.close()

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    # Restore normalizer to training mode
    if obs_normalizer is not None:
        obs_normalizer.set_training_mode(True)
        
    return mean_reward, std_reward

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent on humanoid-walk environment')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='final', 
                      help='Checkpoint to load (e.g. "100000" or "final")')
    parser.add_argument('--env_name', type=str, default=config.ENV_NAME,
                      help='Environment name for loading checkpoint')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # --- Initialization ---
    # Create model save directory (and parent directories if needed)
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    if config.LOG_DIR and not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR, exist_ok=True)

    # Set up CSV logging
    training_log_fields = [
        'episode', 'timestep', 'episode_reward', 'episode_steps', 'avg_reward_100',
        'alpha', 'q1_loss', 'q2_loss', 'actor_loss', 'alpha_loss',
        'log_probs_mean', 'log_probs_min', 'log_probs_max',
        'mu_mean', 'mu_min', 'mu_max',
        'log_std_mean', 'log_std_min', 'log_std_max'
    ]
    training_log_file, training_log_writer = setup_csv_logging(
        config.LOG_DIR, f"{config.ENV_NAME}_training_log.csv", training_log_fields
    )

    # Set up a separate CSV file for alpha debugging
    alpha_debug_fields = [
        'timestep', 'log_probs_mean', 'log_probs_min', 'log_probs_max',
        'target_entropy', 'log_alpha', 'alpha', 'entropy_diff', 'alpha_loss',
        'mu_mean', 'mu_min', 'mu_max',
        'log_std_mean', 'log_std_min', 'log_std_max'
    ]
    alpha_debug_file, alpha_debug_writer = setup_csv_logging(
        config.LOG_DIR, f"{config.ENV_NAME}_alpha_debug.csv", alpha_debug_fields
    )

    # Optional: TensorBoard writer
    # writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Set random seed
    if hasattr(config, 'SEED'):
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        seed = config.SEED
    else:
        seed = np.random.randint(0, 1000000)

    # Create the environment
    env = make_dmc_env(env_name=config.ENV_NAME, seed=seed, flatten=True, use_pixels=False)

    # Get the actual state and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    # Update config if necessary
    if state_dim != config.STATE_DIM:
        print(f"Warning: Actual state dimension ({state_dim}) differs from config ({config.STATE_DIM})")
        config.STATE_DIM = state_dim

    if action_dim != config.ACTION_DIM:
        print(f"Warning: Actual action dimension ({action_dim}) differs from config ({config.ACTION_DIM})")
        config.ACTION_DIM = action_dim

    if action_low != config.ACTION_LOW:
        print(f"Warning: Actual action low ({action_low}) differs from config ({config.ACTION_LOW})")
        config.ACTION_LOW = action_low

    if action_high != config.ACTION_HIGH:
        print(f"Warning: Actual action high ({action_high}) differs from config ({config.ACTION_HIGH})")
        config.ACTION_HIGH = action_high

    # Initialize observation normalizer if enabled
    obs_normalizer = None
    if config.USE_OBS_NORMALIZATION:
        obs_normalizer = ObservationNormalizer(
            state_dim=config.STATE_DIM,
            clip_range=config.OBS_NORM_CLIP,
            device=config.DEVICE
        )

    # Initialize SAC agent
    agent = SAC_agent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        hidden_dim=config.NET_WIDTH,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        alpha_lr=config.ALPHA_LR,
        gamma=config.GAMMA,
        tau=config.TAU,
        alpha=config.ALPHA,
        target_entropy=config.TARGET_ENTROPY,
        auto_entropy_tuning=config.AUTO_ENTROPY_TUNING,
        device=config.DEVICE,
        actor_hidden_layers=config.ACTOR_HIDDEN_LAYERS,
        critic_hidden_layers=config.CRITIC_HIDDEN_LAYERS
    )
    
    # If resuming from checkpoint, load the agent's state
    start_timestep = 0
    if args.resume:
        checkpoint_timestep = args.checkpoint
        checkpoint_env_name = args.env_name
        try:
            agent.load(checkpoint_env_name, checkpoint_timestep)
            print(f"Successfully loaded model from: {config.MODEL_SAVE_DIR}/{checkpoint_env_name}_sac_*_{checkpoint_timestep}.pth")
            
            # Try to load observation normalizer statistics if enabled
            if obs_normalizer is not None:
                try:
                    norm_path = f"{config.MODEL_SAVE_DIR}/{checkpoint_env_name}_obs_norm_{checkpoint_timestep}.npz"
                    obs_normalizer.load(norm_path)
                    print(f"Loaded observation normalizer from {norm_path}")
                except Exception as e:
                    print(f"Warning: Could not load observation normalizer: {e}")
            
            # If checkpoint is a number, use it as starting timestep
            if checkpoint_timestep.isdigit():
                start_timestep = int(checkpoint_timestep)
                print(f"Resuming training from timestep {start_timestep}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.REPLAY_BUFFER_SIZE,
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        device=config.DEVICE
    )

    # --- Training loop ---
    timestep = start_timestep
    episode_num = 0
    episode_rewards_deque = deque(maxlen=100)  # For logging average reward

    # Reset the environment
    # The environment returns tuple (observation, info) in Gymnasium format
    (obs_raw, _) = env.reset(seed=seed)

    # Normalize observation if enabled
    if obs_normalizer is not None:
        # Update normalizer with initial observation
        obs_normalizer.update(obs_raw)
        obs = obs_normalizer.normalize(obs_raw)
    else:
        obs = obs_raw

    episode_reward = 0
    episode_steps = 0

    # Create a progress bar for the total training timesteps
    pbar = tqdm(total=config.MAX_TRAIN_TIMESTEPS, desc="Training Progress")
    pbar.update(timestep)  # Initialize progress bar at current timestep

    # Training metrics
    training_metrics = {
        'q1_loss': 0,
        'q2_loss': 0,
        'actor_loss': 0,
        'alpha_loss': 0,
        'alpha': agent.alpha,
        'log_probs_mean': 0.0,
        'log_probs_min': 0.0,
        'log_probs_max': 0.0,
        'entropy_diff': 0.0,
        'log_alpha': np.log(agent.alpha),
        'target_entropy': agent.target_entropy,
        # Add mu and log_std statistics
        'mu_mean': 0.0,
        'mu_min': 0.0, 
        'mu_max': 0.0,
        'log_std_mean': 0.0,
        'log_std_min': 0.0,
        'log_std_max': 0.0
    }

    # Start time for calculating training speed
    start_time = time.time()

    while timestep < config.MAX_TRAIN_TIMESTEPS:
        # Select action
        if timestep < config.RANDOM_STEPS and not args.resume:
            # Random exploration in the beginning (skip if resuming training)
            action = env.action_space.sample()
        else:
            # Use the agent to select action
            action = agent.select_action(obs, deterministic=False)

        # Take a step in the environment
        # The environment returns (observation, reward, terminated, truncated, info) in Gymnasium format
        (next_obs_raw, reward, terminated, truncated, _) = env.step(action)

        # Debug: Print normalized observation mean/std every 1000 steps
        if obs_normalizer is not None and timestep % 1000 == 0:
            norm_obs = obs_normalizer.normalize(next_obs_raw)
            print(f"[DEBUG] Step {timestep}: Normalized obs mean={np.mean(norm_obs):.4f}, std={np.std(norm_obs):.4f}")

        # Scale reward if needed
        reward = reward * config.REWARD_SCALE

        # Update episode steps
        episode_steps += 1

        # Check if episode is done
        done = terminated or truncated

        # For DeepMind Control environments wrapped in Gymnasium:
        # - terminated: episode ended due to agent reaching terminal state (success or failure)
        # - truncated: episode ended due to time limit or other external factors

        # real_done should be True only for 'terminated' states (natural termination)
        # not for 'truncated' states (timeout)
        real_done = terminated  # True for actual termination, False for timeout

        # Update observation normalizer if enabled
        if obs_normalizer is not None:
            obs_normalizer.update(next_obs_raw)
            next_obs_normalized = obs_normalizer.normalize(next_obs_raw)
        else:
            next_obs_normalized = next_obs_raw

        # Store transition in replay buffer
        # - done: whether episode ended for any reason (terminal state or timeout)
        # - real_done: whether episode ended due to a terminal state (not timeout)
        # This distinction helps with proper TD target calculation
        replay_buffer.add(obs, action, reward, next_obs_normalized, float(done), float(real_done))

        # Update current observation and episode statistics
        obs = next_obs_normalized
        episode_reward += reward
        # episode_steps is already incremented after env.step()
        timestep += 1
        pbar.update(1)  # Update progress bar

        # Train the agent if enough samples are available
        if (timestep >= config.RANDOM_STEPS or args.resume) and len(replay_buffer) > config.BATCH_SIZE:
            # Update the agent
            metrics = agent.train(replay_buffer, config.BATCH_SIZE)

            # Update training metrics (exponential moving average)
            for k, v in metrics.items():
                if k in training_metrics:
                    training_metrics[k] = 0.95 * training_metrics[k] + 0.05 * v

            # Log alpha debugging information every 100 steps
            if timestep % 100 == 0:
                # Print raw metrics for debugging
                print(f"[DEBUG] Raw metrics from agent.train(): {metrics}")

                # Create a dictionary with the metrics
                debug_data = {
                    'timestep': timestep,
                    'log_probs_mean': metrics.get('log_probs_mean', 0.0),
                    'log_probs_min': metrics.get('log_probs_min', 0.0),
                    'log_probs_max': metrics.get('log_probs_max', 0.0),
                    'target_entropy': metrics.get('target_entropy', 0.0),
                    'alpha': metrics.get('alpha', 0.0),
                    'entropy_diff': metrics.get('entropy_diff', 0.0),
                    'alpha_loss': metrics.get('alpha_loss', 0.0),
                    'mu_mean': metrics.get('mu_mean', 0.0),
                    'mu_min': metrics.get('mu_min', 0.0),
                    'mu_max': metrics.get('mu_max', 0.0),
                    'log_std_mean': metrics.get('log_std_mean', 0.0),
                    'log_std_min': metrics.get('log_std_min', 0.0),
                    'log_std_max': metrics.get('log_std_max', 0.0)
                }

                # Add log_alpha directly from agent for consistency
                if agent.auto_entropy_tuning:
                    debug_data['log_alpha'] = metrics.get('log_alpha', 0.0)  # This should always be present
                else:
                    debug_data['log_alpha'] = np.log(agent.alpha)  # Use log of agent's fixed alpha

                alpha_debug_writer.writerow(debug_data)
                alpha_debug_file.flush()  # Ensure data is written to disk

        # End of episode handling
        if done or episode_steps >= config.MAX_EPISODE_STEPS:
            episode_num += 1
            episode_rewards_deque.append(episode_reward)
            avg_reward_last_100 = np.mean(episode_rewards_deque) if episode_rewards_deque else -np.inf

            # Calculate training speed
            elapsed_time = time.time() - start_time
            steps_per_second = timestep / elapsed_time if elapsed_time > 0 else 0

            # Print episode summary
            print(f"Episode: {episode_num}, Steps: {episode_steps}, Reward: {episode_reward:.2f}, "
                  f"AvgReward100: {avg_reward_last_100:.2f}, Alpha: {agent.alpha:.4f}, "
                  f"Steps/sec: {steps_per_second:.2f}")

            # Log to CSV file
            print(f"[CSV DEBUG] Writing to CSV - log_std stats: mean={training_metrics.get('log_std_mean', 0.0):.4f}, min={training_metrics.get('log_std_min', 0.0):.4f}, max={training_metrics.get('log_std_max', 0.0):.4f}")
            training_log_writer.writerow({
                'episode': episode_num,
                'timestep': timestep,
                'episode_reward': episode_reward,
                'episode_steps': episode_steps,
                'avg_reward_100': avg_reward_last_100,
                'alpha': agent.alpha,
                'q1_loss': training_metrics.get('q1_loss', 0.0),
                'q2_loss': training_metrics.get('q2_loss', 0.0),
                'actor_loss': training_metrics.get('actor_loss', 0.0),
                'alpha_loss': training_metrics.get('alpha_loss', 0.0),
                'log_probs_mean': training_metrics.get('log_probs_mean', 0.0),
                'log_probs_min': training_metrics.get('log_probs_min', 0.0),
                'log_probs_max': training_metrics.get('log_probs_max', 0.0),
                'mu_mean': training_metrics.get('mu_mean', 0.0),
                'mu_min': training_metrics.get('mu_min', 0.0),
                'mu_max': training_metrics.get('mu_max', 0.0),
                'log_std_mean': training_metrics.get('log_std_mean', 0.0),
                'log_std_min': training_metrics.get('log_std_min', 0.0),
                'log_std_max': training_metrics.get('log_std_max', 0.0)
            })
            training_log_file.flush()  # Ensure data is written to disk

            # Optional: Log to TensorBoard
            # writer.add_scalar('rollout/ep_reward', episode_reward, timestep)
            # writer.add_scalar('rollout/ep_len', episode_steps, timestep)
            # writer.add_scalar('rollout/avg_reward_100', avg_reward_last_100, timestep)
            # writer.add_scalar('train/alpha', agent.alpha, timestep)
            # for k, v in training_metrics.items():
            #     writer.add_scalar(f'train/{k}', v, timestep)

            # Reset the environment
            # The environment returns tuple (observation, info) in Gymnasium format
            (obs_raw, _) = env.reset(seed=seed + episode_num)

            # Normalize observation if enabled
            if obs_normalizer is not None:
                # Update normalizer with raw observation using a running average approach
                # for more stable statistics
                obs_normalizer.update(obs_raw)
                obs = obs_normalizer.normalize(obs_raw)
            else:
                obs = obs_raw

            episode_reward = 0
            episode_steps = 0

        # Evaluation and model saving
        if timestep % config.EVAL_FREQ == 0 and timestep > 0:
            eval_mean_reward, eval_std_reward = evaluate_agent(
                config.ENV_NAME, agent, obs_normalizer, n_episodes=10, seed=seed+1000
            )
            score = eval_mean_reward - eval_std_reward

            print(f"--------------------------------------------------------")
            print(f"Evaluation at Timestep: {timestep}")
            print(f"Mean Reward: {eval_mean_reward:.2f} +/- {eval_std_reward:.2f}")
            print(f"Score: {score:.2f}")
            print(f"--------------------------------------------------------")

            # Optional: Log to TensorBoard
            # writer.add_scalar('eval/mean_reward', eval_mean_reward, timestep)
            # writer.add_scalar('eval/std_reward', eval_std_reward, timestep)
            # writer.add_scalar('eval/score', score, timestep)

        if timestep % config.SAVE_FREQ == 0 and timestep > 0:
            agent.save(config.ENV_NAME, str(timestep))
            if obs_normalizer is not None:
                obs_normalizer.save(f"{config.MODEL_SAVE_DIR}/{config.ENV_NAME}_obs_norm_{timestep}.npz")
            print(f"Saved model at timestep {timestep}")

    # Save final model
    agent.save(config.ENV_NAME, "final")
    if obs_normalizer is not None:
        obs_normalizer.save(f"{config.MODEL_SAVE_DIR}/{config.ENV_NAME}_obs_norm_final.npz")

    print("Training finished. Saved final model.")
    pbar.close()  # Close the progress bar
    env.close()

    # Close CSV log files
    training_log_file.close()
    alpha_debug_file.close()
    print(f"Training logs saved to {config.LOG_DIR}")

    # writer.close()

if __name__ == '__main__':
    # Example usage:
    # 1. Start new training: python train.py
    # 2. Resume from checkpoint: python train.py --resume --checkpoint 100000
    # 3. Resume from final model: python train.py --resume --checkpoint final
    main()
