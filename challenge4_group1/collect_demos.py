import numpy as np
import torch

from challenge3_group1.env_utils import make_env
from challenge3_group1.model import AtariActorCritic, preprocess_obs


def collect_demonstrations(
    checkpoint_path: str,
    output_path: str,
    n_steps: int,
    device: str = "cpu",
):

    env = make_env()

    n_actions = env.action_space.n

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = AtariActorCritic(n_actions).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    obs_buffer = []
    action_buffer = []

    obs, _ = env.reset()

    for _ in range(n_steps):

        obs_t = preprocess_obs(obs, device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(obs_t)

        action = logits.argmax(dim=-1).item()

        obs_buffer.append(
            np.array(obs, dtype=np.float32)
        )

        action_buffer.append(action)

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    np.savez_compressed(
        output_path,
        observations=np.array(obs_buffer, dtype=np.float32),
        actions=np.array(action_buffer, dtype=np.int64),
    )

    print(f"Saved {n_steps} demonstrations -> {output_path}")