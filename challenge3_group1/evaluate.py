"""
evaluate.py
-----------
Load a trained PPO checkpoint and evaluate it on ALE/MontezumaRevenge-v5.

Usage:
    # Evaluate best model from a specific run:
    python evaluate.py --checkpoint logs/montezuma_ppo/<run_name>/checkpoints/best_model.pt

    # Evaluate with rendering (watch the agent play):
    python evaluate.py --checkpoint <path> --render

    # Evaluate with more episodes:
    python evaluate.py --checkpoint <path> --n_episodes 20

    # Save a returns summary to a file:
    python evaluate.py --checkpoint <path> --save_results
"""

import argparse
import os

import numpy as np
import torch

from env_utils import make_env, ENV_ID
from model import AtariActorCritic, preprocess_obs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    n_episodes: int = 10,
    render: bool = False,
    seed: int = 100,
    save_results: bool = False,
):
    """
    Load a checkpoint and run n_episodes with greedy policy.

    Args:
        checkpoint_path : Path to .pt checkpoint file.
        n_episodes      : Number of evaluation episodes.
        render          : If True, open a window to watch the agent.
        seed            : Base seed for evaluation environments.
        save_results    : If True, save returns array next to checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")

    # --- Load checkpoint ---
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("cfg", {})
    step = ckpt.get("step", "unknown")

    print(f"Trained for: {step:,} steps" if isinstance(step, int) else f"Trained for: {step} steps")
    print(f"Config     : {cfg}\n")

    # --- Build model ---
    render_mode = "human" if render else None
    env = make_env(seed=seed, render_mode=render_mode)
    n_actions = env.action_space.n

    model = AtariActorCritic(n_actions).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Run episodes ---
    returns = []
    rooms_visited = []  # Track rooms entered (Montezuma specific metric)

    print(f"Running {n_episodes} evaluation episodes (greedy policy)...\n")

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, info = env.reset()
            ep_return = 0.0
            ep_steps  = 0
            done = False
            max_room = 0

            while not done:
                obs_t  = preprocess_obs(obs, device).unsqueeze(0)
                logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()  # greedy — no sampling

                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                ep_steps  += 1
                done = terminated or truncated

                # Montezuma-specific: track room number if available
                if "ram" in info:
                    # RAM address 3 = current room in Montezuma's Revenge
                    room = int(info["ram"][3])
                    max_room = max(max_room, room)

            returns.append(ep_return)
            rooms_visited.append(max_room)

            print(
                f"  Episode {ep+1:>3}/{n_episodes} | "
                f"return={ep_return:>7.1f} | "
                f"steps={ep_steps:>5}"
            )

    env.close()

    # --- Summary ---
    mean_r = np.mean(returns)
    std_r  = np.std(returns)
    min_r  = np.min(returns)
    max_r  = np.max(returns)

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY ({n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean return : {mean_r:.2f}")
    print(f"  Std  return : {std_r:.2f}")
    print(f"  Min  return : {min_r:.2f}")
    print(f"  Max  return : {max_r:.2f}")
    print(f"  Non-zero episodes: {sum(r > 0 for r in returns)}/{n_episodes}")
    print(f"{'='*50}\n")

    # --- Save results ---
    if save_results:
        out_dir = os.path.dirname(checkpoint_path)
        out_path = os.path.join(out_dir, "eval_returns.npy")
        np.save(out_path, np.array(returns))
        print(f"Returns saved to: {out_path}")

    return mean_r, std_r, returns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO agent on Montezuma's Revenge")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt checkpoint file (best_model.pt or final_model.pt)"
    )
    p.add_argument("--n_episodes",   type=int,  default=10,    help="Number of evaluation episodes")
    p.add_argument("--render",       action="store_true",       help="Render the environment visually")
    p.add_argument("--seed",         type=int,  default=100,   help="Base seed for evaluation")
    p.add_argument("--save_results", action="store_true",       help="Save returns array to disk")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path = args.checkpoint,
        n_episodes      = args.n_episodes,
        render          = args.render,
        seed            = args.seed,
        save_results    = args.save_results,
    )