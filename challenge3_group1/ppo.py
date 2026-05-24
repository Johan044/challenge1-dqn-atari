import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation (GAE)
# Schulman et al., 2016 — https://arxiv.org/abs/1506.02438
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list,
    values: list,
    dones: list,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and discounted returns for a rollout buffer.

    Args:
        rewards    : List of per-step rewards collected during rollout.
        values     : List of V(s_t) estimates from the critic (torch scalars).
        dones      : List of done flags (True if episode ended at step t).
        next_value : V(s_{T+1}) — value of the state after the last rollout step.
        gamma      : Discount factor.
        gae_lambda : GAE lambda parameter (trade-off bias/variance).
        device     : Torch device for output tensors.

    Returns:
        advantages : GAE advantage estimates, shape (T,).
        returns    : Discounted returns (advantages + values), shape (T,).

    GAE formula:
        delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t       = delta_t + (gamma * lambda) * A_{t+1} * (1 - done_t)
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=device)

    # Convert value list to a plain tensor for indexing
    values_tensor = torch.stack(values).to(device)  # shape (T,)

    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = next_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = values_tensor[t + 1].item()

        delta = rewards[t] + gamma * next_val * next_non_terminal - values_tensor[t].item()
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values_tensor
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Update Step
# Schulman et al., 2017 — https://arxiv.org/abs/1707.06347
# ---------------------------------------------------------------------------

def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_buf: torch.Tensor,
    act_buf: torch.Tensor,
    logp_old_buf: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    n_epochs: int = 4,
    batch_size: int = 64,
    clip_eps: float = 0.1,
    vf_coef: float = 0.5,
    ent_coef: float = 0.02,
    max_grad_norm: float = 0.5,
) -> dict:
    """
    Run K epochs of mini-batch PPO updates over a collected rollout.

    Args:
        model        : AtariActorCritic network.
        optimizer    : Adam optimizer.
        obs_buf      : Observations tensor, shape (T, 4, 84, 84).
        act_buf      : Actions tensor, shape (T,).
        logp_old_buf : Log-probs under old policy, shape (T,).
        advantages   : GAE advantages, shape (T,). Will be normalised inside.
        returns      : Discounted returns, shape (T,).
        n_epochs     : Number of update epochs per rollout (K).
        batch_size   : Mini-batch size.
        clip_eps     : PPO clipping epsilon (epsilon in paper).
        vf_coef      : Value-function loss coefficient (c1).
        ent_coef     : Entropy bonus coefficient (c2).
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Dictionary with mean losses for logging:
            {policy_loss, value_loss, entropy, total_loss, approx_kl, clip_fraction}

    Combined loss (following the paper sign convention):
        L = -L_CLIP + c1 * L_VF - c2 * L_ENT
    """
    T = obs_buf.shape[0]
    device = obs_buf.device

    # Normalise advantages over the full rollout buffer
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Tracking metrics across all epochs and mini-batches
    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "total_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(n_epochs):
        # Shuffle indices each epoch for unbiased mini-batches
        idx = torch.randperm(T, device=device)

        for start in range(0, T, batch_size):
            mb_idx = idx[start: start + batch_size]

            mb_obs  = obs_buf[mb_idx]
            mb_act  = act_buf[mb_idx]
            mb_adv  = adv_norm[mb_idx]
            mb_ret  = returns[mb_idx]
            mb_logp_old = logp_old_buf[mb_idx]

            # Forward pass with current policy
            logits, val_new = model(mb_obs)
            dist_new = Categorical(logits=logits)
            logp_new = dist_new.log_prob(mb_act)
            entropy  = dist_new.entropy().mean()

            # Probability ratio r_t(theta) = pi_new / pi_old
            log_ratio = logp_new - mb_logp_old
            ratio = log_ratio.exp()

            # Approximate KL divergence (for monitoring)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            # Clipped surrogate objective
            surr1 = ratio * mb_adv
            surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            loss_pi = -torch.min(surr1, surr2).mean()

            # Value function loss
            loss_vf = ((val_new - mb_ret) ** 2).mean()

            # Combined loss
            loss = loss_pi + vf_coef * loss_vf - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["policy_loss"].append(loss_pi.item())
            metrics["value_loss"].append(loss_vf.item())
            metrics["entropy"].append(entropy.item())
            metrics["total_loss"].append(loss.item())
            metrics["approx_kl"].append(approx_kl)
            metrics["clip_fraction"].append(clip_frac)

    # Return means for logging
    return {k: sum(v) / len(v) for k, v in metrics.items()}


if __name__ == "__main__":
    """Quick sanity check: one PPO update on random data."""
    import sys
    sys.path.insert(0, ".")
    from model import AtariActorCritic

    device = torch.device("cpu")
    T = 1024  # rollout horizon
    n_actions = 18

    model = AtariActorCritic(n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    # Fake rollout data
    obs_buf  = torch.rand(T, 4, 84, 84, device=device)
    act_buf  = torch.randint(0, n_actions, (T,), device=device)
    logp_buf = torch.randn(T, device=device)
    rewards  = [0.0] * T
    values   = [torch.tensor(0.0)] * T
    dones    = [False] * T

    advantages, returns = compute_gae(
        rewards, values, dones,
        next_value=0.0, gamma=0.99, gae_lambda=0.95, device=device
    )

    print(f"Advantages shape : {advantages.shape}")
    print(f"Returns shape    : {returns.shape}")

    metrics = ppo_update(
        model, optimizer,
        obs_buf, act_buf, logp_buf,
        advantages, returns,
        n_epochs=2, batch_size=64,
    )

    print("\nPPO update metrics:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.6f}")

    print("\nppo.py OK")