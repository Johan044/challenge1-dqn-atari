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
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnvWrapper
from collections import deque

ENV_ID = "ALE/MontezumaRevenge-v5"
N_STACK = 4


# ================================
# Exploration reward wrapper
# ================================

class ExplorationRewardWrapper(VecEnvWrapper):
  

    def __init__(self, venv, bonus=0.1):
        super().__init__(venv)
        self.bonus = bonus
        self.visited = [set() for _ in range(venv.num_envs)]

    def reset(self):
        obs = self.venv.reset()
        self.visited = [set() for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        new_rewards = []

        for i in range(self.num_envs):
            last_frame = obs[i][:, :, -1] if obs[i].ndim == 3 else obs[i][..., -1]
            small = last_frame[::4, ::4]
            state_hash = hash(small.tobytes())

            bonus = 0.0
            if state_hash not in self.visited[i]:
                self.visited[i].add(state_hash)
                bonus = self.bonus

            if dones[i]:
                self.visited[i].clear()

            new_rewards.append(rewards[i] + bonus)

        return obs, np.array(new_rewards), dones, infos


# ================================
# Tensorboard logging
# ================================

class TensorBoardCallback(BaseCallback):
    def __init__(self, seed=42):
        super().__init__()
        self.writer = None
        self.episode_reward = 0
        self.episode_length = 0
        self.seed = seed
        self.episode_rewards_history = deque(maxlen=100)

    def _on_training_start(self):
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self.writer = fmt.writer
                # Loguea el seed usado para reproducibilidad
                self.writer.add_text("config/seed", str(self.seed), 0)
                return

    def _on_step(self):
        if self.writer is None:
            return True

        reward = float(self.locals["rewards"][0])
        self.episode_reward += reward
        self.episode_length += 1

        # Epsilon
        self.writer.add_scalar(
            "training/epsilon",
            self.model.exploration_rate,
            self.num_timesteps
        )

        # Loss (disponible después de learning_starts)
        if self.model.logger and hasattr(self.model, '_logger'):
            loss = self.model._logger.name_to_value.get("train/loss", None)
            if loss is not None:
                self.writer.add_scalar("training/loss", loss, self.num_timesteps)

        if self.locals["dones"][0]:
            self.episode_rewards_history.append(self.episode_reward)

            self.writer.add_scalar(
                "training/episode_reward",
                self.episode_reward,
                self.num_timesteps
            )

            self.writer.add_scalar(
                "training/episode_length",
                self.episode_length,
                self.num_timesteps
            )

            # Rolling mean (ventana 100 episodios — requerido por el taller)
            if len(self.episode_rewards_history) >= 10:
                rolling_mean = np.mean(self.episode_rewards_history)
                self.writer.add_scalar(
                    "training/rolling_mean_reward_100",
                    rolling_mean,
                    self.num_timesteps
                )

            self.episode_reward = 0
            self.episode_length = 0

        return True


# ================================
# ENVIRONMENT BUILDERS
# ================================

def build_training_environment(seed):
    env = make_atari_env(
        ENV_ID,
        n_envs=8,
        seed=seed
    )
    env = VecFrameStack(env, n_stack=N_STACK)
    env = ExplorationRewardWrapper(env)
    return env


def build_playing_environment():
    def make_env():
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(
            base_env,
            terminal_on_life_loss=True,
            clip_reward=False
        )
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# ================================
# TRAINING FUNCTION
# ================================

def train_agent(model_path, timesteps, seed, tensorboard_log, hparams=None):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if hparams is None:
        # Defaults alineados con el rango del taller
        hparams = dict(
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=20000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=2000,
            exploration_fraction=0.25,
            exploration_final_eps=0.01
        )

    # Loguea hparams en tensorboard
    writer = SummaryWriter(log_dir=tensorboard_log)
    hparams_to_log = {k: v for k, v in hparams.items() if k != "name"}
    hparams_to_log["seed"] = seed
    writer.add_hparams(hparams_to_log, {"hparam/reward": 0})
    writer.close()

    env = build_training_environment(seed)

    if os.path.exists(f"{model_path}.zip"):
        print(f"Cargando modelo existente: {model_path}.zip")
        model = DQN.load(model_path, env=env)
    else:
        print(f"Creando nuevo modelo con hparams: {hparams_to_log}")
        model = DQN(
            "CnnPolicy",
            env,
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
            seed=seed
        )

    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(seed=seed),
        progress_bar=True,
        tb_log_name=Path(model_path).name
    )

    model.save(model_path)
    env.close()

    print(f"Modelo guardado en {model_path}.zip")

    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0


# ================================
# PLAY
# ================================

def play_agent(model_path, episodes):
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}.zip")

    env = build_playing_environment()
    model = DQN.load(model_path, env=env)

    completed = 0
    obs = env.reset()
    episode_reward = 0

    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episodio {completed} — reward: {episode_reward:.1f}")
                episode_reward = 0

    env.close()


# ================================
# ARGUMENT PARSER
# ================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "play", "sweep"],
        required=True
    )
    parser.add_argument("--model-path", default="models/montezuma_dqn")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard-log", default="logs/montezuma_dqn")

    # NUEVO: permite cargar hparams del JSON por nombre de experimento
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Nombre del experimento en sweep_configs.json (ej: exp_07_more_exploration)"
    )
    parser.add_argument(
        "--sweep-file",
        default="sweep_configs.json",
        help="Ruta al archivo JSON de configuraciones"
    )

    return parser.parse_args()


# ================================
# MAIN
# ================================

def main():
    args = parse_args()

    if args.mode == "train":

        hparams = None

        # Si se pasa --exp-name, carga los hparams del JSON
        if args.exp_name:
            if not os.path.exists(args.sweep_file):
                print(f"ERROR: No se encontró {args.sweep_file}")
                return
            with open(args.sweep_file) as f:
                sweeps = json.load(f)
            match = next((s for s in sweeps if s["name"] == args.exp_name), None)
            if match is None:
                print(f"ERROR: Experimento '{args.exp_name}' no encontrado en {args.sweep_file}")
                print(f"Experimentos disponibles: {[s['name'] for s in sweeps]}")
                return
            hparams = match
            print(f"Usando hparams de '{args.exp_name}' del JSON")

        train_agent(
            model_path=args.model_path,
            timesteps=args.timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
            hparams=hparams
        )

    elif args.mode == "play":
        play_agent(
            model_path=args.model_path,
            episodes=args.episodes
        )

    elif args.mode == "sweep":
        if not os.path.exists(args.sweep_file):
            print(f"ERROR: No se encontró {args.sweep_file}")
            return

        with open(args.sweep_file) as f:
            sweeps = json.load(f)

        for sweep in sweeps:
            print(f"\n=== Entrenando: {sweep['name']} ===")
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