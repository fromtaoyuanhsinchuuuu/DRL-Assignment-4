# train.py
import gymnasium as gym
import numpy as np
import torch
import os
from collections import deque  # For storing recent episode rewards for logging
from tqdm import tqdm  # For progress bar

import config  # Import hyperparameters
from ppo_core import PPO_agent  # Import PPO core class
# Optional: from torch.utils.tensorboard import SummaryWriter

def evaluate_agent(env, agent, n_episodes=10):
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            # action should already be in env range from select_action
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return mean_reward, std_reward

def main():
    # --- Initialization ---
    if not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
    if config.LOG_DIR and not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)

    # Optional: TensorBoard writer
    # writer = SummaryWriter(log_dir=config.LOG_DIR)

    env = gym.make(config.ENV_NAME)
    # eval_env = gym.make(config.ENV_NAME)  # For evaluation

    # Set random seed (optional)
    if hasattr(config, 'SEED'):
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        # env.seed(config.SEED)  # gym 0.26+ uses env.reset(seed=SEED)
        # eval_env.seed(config.SEED + 1)

    agent = PPO_agent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        net_width=config.NET_WIDTH,
        dvc=config.DEVICE,
        distribution_type=config.DISTRIBUTION_TYPE,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
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
        action_low=config.ACTION_LOW,
        action_high=config.ACTION_HIGH
    )

    # --- Training loop ---
    timestep = 0
    episode_num = 0
    episode_rewards_deque = deque(maxlen=100)  # For logging average reward

    # Initialize trajectory storage
    # These will be filled up to T_HORIZON steps
    s_hoder = np.zeros((config.T_HORIZON, config.STATE_DIM), dtype=np.float32)
    a_hoder = np.zeros((config.T_HORIZON, config.ACTION_DIM), dtype=np.float32)  # Store action in model's native range if log_prob is for it
    r_hoder = np.zeros((config.T_HORIZON, 1), dtype=np.float32)
    s_next_hoder = np.zeros((config.T_HORIZON, config.STATE_DIM), dtype=np.float32)
    # logprob_a_hoder stores log_prob of the action 'a_hoder' under the policy that generated it
    logprob_a_hoder = np.zeros((config.T_HORIZON, config.ACTION_DIM if config.ACTION_DIM > 1 and config.DISTRIBUTION_TYPE != 'Beta' else 1), dtype=np.float32)  # Beta log_prob is scalar per action
    done_hoder = np.zeros((config.T_HORIZON, 1), dtype=bool)
    dw_hoder = np.zeros((config.T_HORIZON, 1), dtype=bool)  # Done and Well (not truncated)

    obs, _ = env.reset(seed=config.SEED if hasattr(config, 'SEED') else None)
    episode_reward = 0
    episode_steps = 0
    episode_num = 0

    # Create a progress bar for the total training timesteps
    pbar = tqdm(total=config.MAX_TRAIN_TIMESTEPS, desc="Training Progress")
    pbar.update(0)  # Initialize progress bar at 0

    while timestep < config.MAX_TRAIN_TIMESTEPS:
        for t_horizon_step in range(config.T_HORIZON):
            # select_action now returns:
            # 1. action_env: The action to use with env.step()
            # 2. action_model: The action in model's native range that log_prob corresponds to
            # 3. log_prob: The log_prob of action_model
            action_env, action_model, log_prob_action = agent.select_action(obs, deterministic=False)

            # Use action_env for environment interaction
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated
            dw = terminated and not truncated  # Done and Well: episode ended naturally, not due to time limit

            s_hoder[t_horizon_step] = obs

            # IMPORTANT: Store action_model (the model's native action) in a_hoder
            # This is the action that log_prob_action corresponds to
            a_hoder[t_horizon_step] = action_model.reshape(config.ACTION_DIM) if isinstance(action_model, np.ndarray) else [action_model]

            r_hoder[t_horizon_step] = [reward]
            s_next_hoder[t_horizon_step] = next_obs
            logprob_a_hoder[t_horizon_step] = log_prob_action.reshape(config.ACTION_DIM if config.ACTION_DIM > 1 and config.DISTRIBUTION_TYPE != 'Beta' else 1)
            done_hoder[t_horizon_step] = [done]
            dw_hoder[t_horizon_step] = [dw]

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            timestep += 1
            pbar.update(1)  # Update progress bar by 1 step

            if done:
                episode_num += 1
                episode_rewards_deque.append(episode_reward)
                avg_reward_last_100 = np.mean(episode_rewards_deque) if episode_rewards_deque else -np.inf
                print(f"Timestep: {timestep}, Episode: {episode_num}, Reward: {episode_reward:.2f}, AvgReward100: {avg_reward_last_100:.2f}, Epsilon (EntropyCoef): {agent.entropy_coef:.4f}")
                # writer.add_scalar('rollout/ep_reward', episode_reward, timestep)
                # writer.add_scalar('rollout/ep_len', episode_steps, timestep)
                # writer.add_scalar('rollout/avg_reward_100', avg_reward_last_100, timestep)
                # writer.add_scalar('train/entropy_coef', agent.entropy_coef, timestep)

                obs, _ = env.reset(seed=(config.SEED + episode_num) if hasattr(config, 'SEED') else None)

                episode_reward = 0
                episode_steps = 0

            if timestep % config.EVAL_FREQ == 0:
                eval_mean_reward, eval_std_reward = evaluate_agent(gym.make(config.ENV_NAME, render_mode=None), agent, n_episodes=10)  # Use a fresh env for eval
                print(f"--------------------------------------------------------")
                print(f"Evaluation at Timestep: {timestep}")
                print(f"Mean Reward: {eval_mean_reward:.2f} +/- {eval_std_reward:.2f}")
                print(f"--------------------------------------------------------")
                # writer.add_scalar('eval/mean_reward', eval_mean_reward, timestep)
                # writer.add_scalar('eval/std_reward', eval_std_reward, timestep)

            if timestep % config.SAVE_FREQ == 0:
                agent.save(config.ENV_NAME, str(timestep))
                print(f"Saved model at timestep {timestep}")

            if timestep >= config.MAX_TRAIN_TIMESTEPS:
                break
        # End of T_HORIZON loop

        if timestep >= config.MAX_TRAIN_TIMESTEPS:
            break

        # Prepare data for agent.train()
        trajectory_data = {
            's': torch.from_numpy(s_hoder),
            'a': torch.from_numpy(a_hoder),
            'r': torch.from_numpy(r_hoder),
            's_next': torch.from_numpy(s_next_hoder),
            'logprob_a': torch.from_numpy(logprob_a_hoder),
            'done': torch.from_numpy(done_hoder),
            'dw': torch.from_numpy(dw_hoder)
        }
        agent.train(trajectory_data)

    agent.save(config.ENV_NAME, "final")
    print("Training finished. Saved final model.")
    pbar.close()  # Close the progress bar
    env.close()
    # eval_env.close()
    # writer.close()

if __name__ == '__main__':
    main()
