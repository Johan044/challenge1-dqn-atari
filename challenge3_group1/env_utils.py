"""
env_utils.py
------------
Environment factory for ALE/MontezumaRevenge-v5.
Applies standard Atari preprocessing identical to Challenge 1 (DQN)
so that any performance difference is attributable to the algorithm.

Preprocessing pipeline:
  - AtariPreprocessing: noop_max=30, frame_skip=4, resize 84x84, grayscale, scale [0,1]
  - FrameStackObservation: stack 4 consecutive frames
"""

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
import ale_py

gym.register_envs(ale_py)  # required for ale-py >= 0.11


ENV_ID = "ALE/MontezumaRevenge-v5"


def make_env(env_id: str = ENV_ID, seed: int = 0, render_mode: str = None) -> gym.Env:
    """
    Build a preprocessed ALE environment compatible with PPO.

    Args:
        env_id      : ALE environment identifier.
        seed        : Random seed for reproducibility.
        render_mode : None for training, 'human' or 'rgb_array' for evaluation/recording.

    Returns:
        A fully wrapped gymnasium environment.
    """
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)  # disable default frameskip; AtariPreprocessing handles it

    env = AtariPreprocessing(
        env,
        noop_max=30,          # random no-ops at episode start (up to 30)
        frame_skip=4,         # repeat each action 4 times
        screen_size=84,       # resize to 84x84
        grayscale_obs=True,   # convert to grayscale
        scale_obs=True,       # pixel values in [0, 1]
        grayscale_newaxis=True,
    )

    env = FrameStackObservation(env, 4)  # stack 4 frames -> shape (4, 84, 84)

    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    """Quick sanity check: one random episode."""
    env = make_env(seed=42)
    obs, _ = env.reset()
    print(f"Observation shape : {obs.shape}")   # expected: (4, 84, 84)
    print(f"Action space      : {env.action_space}")
    print(f"Number of actions : {env.action_space.n}")

    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {step} | total reward: {total_reward}")
            break
    else:
        print(f"500 steps completed | accumulated reward: {total_reward}")

    env.close()
    print("env_utils.py OK")