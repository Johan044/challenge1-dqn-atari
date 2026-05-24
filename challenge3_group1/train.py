"""
train.py
--------
Main PPO training loop for ALE/MontezumaRevenge-v5 (Group 1 — Challenge 3).

Usage:
    # Single run with default hyperparameters (starter config from assignment):
    python train.py

    # Single run overriding specific hyperparameters:
    python train.py --lr 1e-4 --horizon 1024 --ent_coef 0.01 --seed 0

    # Full hyperparameter sweep (reads sweep_configs.json):
    python train.py --sweep

Outputs (written to logs/montezuma_ppo/<run_name>/):
    - TensorBoard event file
    - checkpoints/best_model.pt   (best mean reward so far)
    - checkpoints/final_model.pt  (end of training)
    - returns.npy                 (per-episode returns array)
"""

import argparse
import json
import os
import random
import time

# Base directory: always relative to this file, not the working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs", "montezuma_ppo")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env_utils import make_env, ENV_ID
from model import AtariActorCritic, preprocess_obs
from ppo import compute_gae, ppo_update


# ---------------------------------------------------------------------------
# Default hyperparameters (Group 1 starter config from assignment doc)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    total_steps   = 5_000_000,
    horizon       = 2048,
    n_epochs      = 4,
    batch_size    = 64,
    lr            = 2.5e-4,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_eps      = 0.1,
    ent_coef      = 0.02,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    seed          = 42,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_name(cfg: dict) -> str:
    """Generate a readable run name from key hyperparameters."""
    return (
        f"lr{cfg['lr']}_h{cfg['horizon']}_ep{cfg['n_epochs']}"
        f"_bs{cfg['batch_size']}_ent{cfg['ent_coef']}_seed{cfg['seed']}"
    )


def evaluate_agent(model, device, n_episodes: int = 10, seed: int = 0) -> tuple[float, float]:
    """
    Run n_episodes with greedy policy (no sampling).
    Returns (mean_return, std_return).
    """
    model.eval()
    env = make_env(seed=seed + 9999)  # different seed from training env
    returns = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            done = False
            while not done:
                obs_t = preprocess_obs(obs, device).unsqueeze(0)
                logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()  # greedy
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)

    env.close()
    model.train()
    return float(np.mean(returns)), float(np.std(returns))


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train(cfg: dict, log_dir: str):
    """
    Train a PPO agent with the given configuration.

    Args:
        cfg     : Hyperparameter dictionary.
        log_dir : Directory for TensorBoard logs and checkpoints.
    """
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Run: {os.path.basename(log_dir)}")
    print(f"Device: {device}  |  Config: {cfg}")
    print(f"{'='*60}\n")

    # --- Environment & model ---
    env = make_env(seed=cfg["seed"])
    n_actions = env.action_space.n
    model = AtariActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)

    # --- Logging ---
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # --- Training state ---
    obs, _ = env.reset()
    all_returns = []
    best_mean_reward = -float("inf")
    global_step = 0
    episode_return = 0.0
    episode_count = 0
    start_time = time.time()

    # Evaluation interval: every 500k steps
    eval_interval = 500_000

    # Rolling window for mean reward logging
    reward_window = []

    print(f"Starting training for {cfg['total_steps']:,} steps...\n")

    while global_step < cfg["total_steps"]:

        # ----------------------------------------------------------------
        # 1. Rollout collection
        # ----------------------------------------------------------------
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []

        for _ in range(cfg["horizon"]):
            obs_t = preprocess_obs(obs, device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            obs_buf.append(preprocess_obs(obs, device))
            act_buf.append(action.squeeze())
            logp_buf.append(log_prob.squeeze())
            val_buf.append(value.squeeze())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rew_buf.append(float(reward))
            done_buf.append(done)
            episode_return += reward
            global_step += 1

            if done:
                episode_count += 1
                all_returns.append(episode_return)
                reward_window.append(episode_return)
                if len(reward_window) > 100:
                    reward_window.pop(0)

                writer.add_scalar("train/episode_return", episode_return, global_step)
                writer.add_scalar("train/episode_count", episode_count, global_step)

                episode_return = 0.0
                obs, _ = env.reset()

        # ----------------------------------------------------------------
        # 2. Compute GAE advantages
        # ----------------------------------------------------------------
        with torch.no_grad():
            obs_t = preprocess_obs(obs, device).unsqueeze(0)
            _, next_val = model(obs_t)
            next_value = next_val.item()

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf,
            next_value=next_value,
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            device=device,
        )

        # ----------------------------------------------------------------
        # 3. PPO update
        # ----------------------------------------------------------------
        obs_tensor  = torch.stack(obs_buf).to(device)
        act_tensor  = torch.stack(act_buf).to(device)
        logp_tensor = torch.stack(logp_buf).to(device)

        metrics = ppo_update(
            model, optimizer,
            obs_tensor, act_tensor, logp_tensor,
            advantages, returns,
            n_epochs      = cfg["n_epochs"],
            batch_size    = cfg["batch_size"],
            clip_eps      = cfg["clip_eps"],
            vf_coef       = cfg["vf_coef"],
            ent_coef      = cfg["ent_coef"],
            max_grad_norm = cfg["max_grad_norm"],
        )

        # ----------------------------------------------------------------
        # 4. Logging
        # ----------------------------------------------------------------
        mean_reward_100 = np.mean(reward_window) if reward_window else 0.0

        writer.add_scalar("train/mean_reward_100ep", mean_reward_100, global_step)
        writer.add_scalar("losses/policy_loss",   metrics["policy_loss"],   global_step)
        writer.add_scalar("losses/value_loss",    metrics["value_loss"],    global_step)
        writer.add_scalar("losses/entropy",       metrics["entropy"],       global_step)
        writer.add_scalar("losses/total_loss",    metrics["total_loss"],    global_step)
        writer.add_scalar("losses/approx_kl",     metrics["approx_kl"],    global_step)
        writer.add_scalar("losses/clip_fraction", metrics["clip_fraction"], global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("train/steps_per_second", sps, global_step)

        # Console log every 10 rollouts
        rollout_num = global_step // cfg["horizon"]
        if rollout_num % 10 == 0:
            print(
                f"step={global_step:>8,} | episodes={episode_count:>5} "
                f"| mean_r100={mean_reward_100:>6.1f} "
                f"| entropy={metrics['entropy']:.3f} "
                f"| kl={metrics['approx_kl']:.4f} "
                f"| sps={sps}"
            )

        # ----------------------------------------------------------------
        # 5. Periodic evaluation (greedy policy)
        # ----------------------------------------------------------------
        if global_step % eval_interval < cfg["horizon"]:
            eval_mean, eval_std = evaluate_agent(
                model, device, n_episodes=10, seed=cfg["seed"]
            )
            writer.add_scalar("eval/mean_return", eval_mean, global_step)
            writer.add_scalar("eval/std_return",  eval_std,  global_step)
            print(f"\n[EVAL] step={global_step:,} | mean={eval_mean:.1f} ± {eval_std:.1f}\n")

            # Save best model
            if eval_mean > best_mean_reward:
                best_mean_reward = eval_mean
                torch.save(
                    {"model_state": model.state_dict(), "cfg": cfg, "step": global_step},
                    os.path.join(log_dir, "checkpoints", "best_model.pt"),
                )

    # ----------------------------------------------------------------
    # 6. Save final model and returns array
    # ----------------------------------------------------------------
    torch.save(
        {"model_state": model.state_dict(), "cfg": cfg, "step": global_step},
        os.path.join(log_dir, "checkpoints", "final_model.pt"),
    )
    np.save(os.path.join(log_dir, "returns.npy"), np.array(all_returns))

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed/3600:.2f}h")
    print(f"Best eval mean reward: {best_mean_reward:.1f}")
    print(f"Final model saved to : {log_dir}/checkpoints/final_model.pt")

    writer.close()
    env.close()
    return best_mean_reward, all_returns


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def run_sweep(sweep_file: str = "sweep_configs.json"):
    """Load sweep configs and run each one sequentially."""
    with open(sweep_file, "r") as f:
        configs = json.load(f)

    print(f"Running sweep: {len(configs)} configurations")
    results = []

    for i, override in enumerate(configs):
        cfg = {**DEFAULTS, **override}
        run_name = make_run_name(cfg)
        log_dir = os.path.join(LOGS_DIR, f"sweep_{i:02d}_{run_name}")

        best_reward, _ = train(cfg, log_dir)
        results.append({"config": cfg, "best_reward": best_reward, "run": run_name})
        print(f"\n[Sweep {i+1}/{len(configs)}] {run_name} -> best_reward={best_reward:.1f}\n")

    # Print summary sorted by best reward
    results.sort(key=lambda x: x["best_reward"], reverse=True)
    print("\n" + "="*60)
    print("SWEEP SUMMARY (sorted by best reward):")
    print("="*60)
    for r in results:
        print(f"  {r['best_reward']:>6.1f}  |  {r['run']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PPO training for ALE/MontezumaRevenge-v5")
    p.add_argument("--sweep",         action="store_true",      help="Run full hyperparameter sweep from sweep_configs.json")
    p.add_argument("--total_steps",   type=int,   default=DEFAULTS["total_steps"])
    p.add_argument("--horizon",       type=int,   default=DEFAULTS["horizon"])
    p.add_argument("--n_epochs",      type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--batch_size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",         type=float, default=DEFAULTS["gamma"])
    p.add_argument("--gae_lambda",    type=float, default=DEFAULTS["gae_lambda"])
    p.add_argument("--clip_eps",      type=float, default=DEFAULTS["clip_eps"])
    p.add_argument("--ent_coef",      type=float, default=DEFAULTS["ent_coef"])
    p.add_argument("--vf_coef",       type=float, default=DEFAULTS["vf_coef"])
    p.add_argument("--max_grad_norm", type=float, default=DEFAULTS["max_grad_norm"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--log_dir",       type=str,   default=None, help="Override log directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep:
        run_sweep("sweep_configs.json")
    else:
        cfg = {k: getattr(args, k) for k in DEFAULTS}
        log_dir = args.log_dir or os.path.join(
            LOGS_DIR, make_run_name(cfg)
        )
        train(cfg, log_dir)