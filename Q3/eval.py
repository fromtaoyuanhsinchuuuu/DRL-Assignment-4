"""
Evaluation script for Humanoid Walk task.
"""

import argparse
import importlib
import numpy as np
from tqdm import tqdm
from dmc import make_dmc_env
import config

def parse_arguments():
    parser = argparse.ArgumentParser(description="DRL HW4 Q3 - Humanoid Walk")
    parser.add_argument("--episodes", default=100, type=int, help="Number of episodes to evaluate")
    parser.add_argument("--record_demo", action="store_true", help="Record a demonstration")
    # Add token parameter for grading system
    parser.add_argument("--token", type=str, help="Evaluation token (used by grading system)")
    return parser.parse_args()

def load_agent(agent_path):
    """Dynamically load the student's agent class"""
    spec = importlib.util.spec_from_file_location("student_agent", agent_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)
    return student_module.Agent()

def record_video(env, agent):
    """Record a video of the agent's performance"""
    import imageio
    gif_path = f'./demo.gif'

    # Reset the environment
    obs, _ = env.reset()
    frames = []

    episode_steps = 0
    while True:
        # Render the environment
        frame = env.render()
        frames.append(np.array(frame))

        # Get action from agent
        action = agent.act(obs)

        # Take a step in the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = next_obs

        episode_steps += 1

        # Check if episode is done
        done = terminated or truncated
        if done or episode_steps >= config.MAX_EPISODE_STEPS:
            break

    imageio.mimsave(gif_path, frames, fps=30)
    print(f'GIF saved to {gif_path}')

def eval_score():
    """Evaluate the agent's performance on Humanoid Walk"""
    args = parse_arguments()

    # Load student's agent
    agent = load_agent("student_agent.py")

    # Run evaluation
    episode_rewards = []

    for episode in tqdm(range(args.episodes), desc="Evaluating"):
        # Create a new environment for each episode
        env = make_dmc_env(env_name=config.ENV_NAME, seed=episode, flatten=True, use_pixels=False)

        # Reset the environment
        obs, _ = env.reset()

        episode_reward = 0
        episode_steps = 0

        while True:
            # Get action from student's agent
            action = agent.act(obs)

            # Take a step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Update observation and reward
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            # Check if episode is done
            done = terminated or truncated
            if done or episode_steps >= config.MAX_EPISODE_STEPS:
                break

        episode_rewards.append(episode_reward)
        env.close()

    # Calculate final score
    mean = np.mean(episode_rewards)
    std = np.std(episode_rewards)
    score = mean - std

    print(f"\nEvaluation complete!")
    print(f"Average return over {args.episodes} episodes: {mean:.2f} (std: {std:.2f})")
    print(f"Final score: {score:.2f}")

    # Record a demonstration if requested
    if args.record_demo:
        env = make_dmc_env(env_name=config.ENV_NAME, seed=0, flatten=True, use_pixels=False)
        record_video(env, agent)
        env.close()

    return np.round(score, 2)

if __name__ == "__main__":
    score = eval_score()

    # Define the baselines
    BASELINE_SCORE = 450

    # Calculate grade percentage
    if score >= BASELINE_SCORE:
        # Beat baseline
        grade_percentage = 20
        result = "EXCELLENT! Beat the baseline"
    else:
        # Did not beat baseline
        # Normalize score between 0 and baseline
        normalized_score = max(0, score) / BASELINE_SCORE
        grade_percentage = normalized_score * 20
        result = "DID NOT meet the baseline"

    print(f"\n{result}")
    print(f"Grade: {grade_percentage:.2f}% out of 20%")
    if grade_percentage >= 20:
        print("\033[92m🌟 CONGRATULATIONS! You did great! 🌟\033[0m")
    else:
        print(f"\033[93mYou earned {grade_percentage:.2f}% out of 20%.\033[0m")

    print("\nNote: There's also a leaderboard component worth an additional 20%.")
