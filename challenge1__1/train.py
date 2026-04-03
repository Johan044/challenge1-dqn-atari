import argparse
import json
import os
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Atari environment
ENV_ID = "ALE/MontezumaRevenge-v5"
N_STACK = 4  # frames stacked


class TensorBoardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.writer = None
        self.episode_reward = 0.0

    def _on_training_start(self):
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self.writer = fmt.writer
                return

    def _on_step(self):
        if self.writer is None:
            return True

        self.episode_reward += float(self.locals["rewards"][0])

        # log exploration rate
        self.writer.add_scalar(
            "training/epsilon",
            self.model.exploration_rate,
            self.num_timesteps
        )

        if self.locals["dones"][0]:
            self.writer.add_scalar(
                "training/episode_reward",
                self.episode_reward,
                self.num_timesteps
            )
            self.episode_reward = 0.0

        return True


def build_training_environment(seed):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def build_playing_environment():
    def make_env():
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(base_env, terminal_on_life_loss=True, clip_reward=False)
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def train_agent(model_path, timesteps, seed, tensorboard_log, hparams=None):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if hparams is None:
        hparams = dict(
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=10000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.15,
            exploration_final_eps=0.01,
        )

    # log hyperparameters
    writer = SummaryWriter(log_dir=tensorboard_log)
    writer.add_hparams(hparams, {"hparam/reward": 0})
    writer.close()

    env = build_training_environment(seed)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=hparams["learning_rate"],
        buffer_size=hparams["buffer_size"],
        learning_starts=hparams["learning_starts"],
        batch_size=hparams["batch_size"],
        gamma=hparams["gamma"],
        train_freq=hparams["train_freq"],
        target_update_interval=hparams["target_update_interval"],
        exploration_fraction=hparams["exploration_fraction"],
        exploration_final_eps=hparams["exploration_final_eps"],
        tensorboard_log=tensorboard_log,
        verbose=1,
        seed=seed,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(),
        progress_bar=True
    )

    model.save(model_path)
    env.close()
    print(f"Model saved at {model_path}.zip")

    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0


def play_agent(model_path, episodes):
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError("Model not found")

    env = build_playing_environment()
    model = DQN.load(model_path, env=env)

    completed = 0
    obs = env.reset()
    episode_reward = 0.0

    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed} reward: {episode_reward}")
                episode_reward = 0.0

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "play", "sweep"],  # <-- agregamos sweep
        required=True
    )
    parser.add_argument("--model-path", default="models/montezuma_dqn")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-log", default="logs/montezuma_dqn")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        train_agent(
            model_path=args.model_path,
            timesteps=args.timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log
        )

    elif args.mode == "play":
        play_agent(
            model_path=args.model_path,
            episodes=args.episodes
        )

    elif args.mode == "sweep":
        # Leemos el JSON
        with open("sweep_configs.json", "r") as f:
            sweeps = json.load(f)

        for sweep in sweeps:
            print(f"\n=== Training sweep: {sweep['name']} ===")
            model_path = f"models/{sweep['name']}"
            tensorboard_log = f"{args.tensorboard_log}/{sweep['name']}"
            train_agent(
                model_path=model_path,
                timesteps=sweep.get("timesteps", 200000),
                seed=args.seed,
                tensorboard_log=tensorboard_log,
                hparams=sweep
            )


if __name__ == "__main__":
    main()