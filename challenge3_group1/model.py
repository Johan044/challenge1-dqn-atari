"""
model.py
--------
Shared convolutional Actor-Critic network for PPO on Atari.

Architecture:
  Input : (batch, 4, 84, 84)  — 4 stacked grayscale frames
  CNN   : 3 convolutional layers (same as Nature DQN)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class AtariActorCritic(nn.Module):
    """
    Shared CNN backbone with separate actor and critic heads.

    The CNN feature extractor is identical to the one used in the
    Nature DQN paper (Mnih et al., 2015), ensuring a fair comparison
    with the DQN agent from Challenge 1.
    """

    def __init__(self, n_actions: int):
        super().__init__()

        # Input: (batch, 4, 84, 84) — 4 stacked greyscale frames
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),                                 # -> 3136
        )

        cnn_out = 64 * 7 * 7  # 3136

        # Actor head: outputs logits for each discrete action
        self.actor = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        # Critic head: outputs scalar state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Orthogonal initialisation — standard practice for PPO
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for conv and linear layers, as recommended for PPO."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Actor output layer: smaller gain for stable initial policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

        # Critic output layer: gain=1.0
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 4, 84, 84), dtype float32, values in [0, 1].

        Returns:
            logits : (batch, n_actions) — unnormalised action scores
            value  : (batch,)           — scalar state value estimates
        """
        feats = self.cnn(x)
        logits = self.actor(feats)
        value = self.critic(feats).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        Sample an action and compute log-prob, entropy, and value.
        Used during rollout collection and PPO update.

        Args:
            x      : Observation tensor (batch, 4, 84, 84).
            action : If provided, compute log-prob of this action (update step).
                     If None, sample a new action (rollout step).

        Returns:
            action   : Sampled or provided action tensor.
            log_prob : Log-probability of the action.
            entropy  : Policy entropy (scalar mean over batch).
            value    : State value estimate.
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return action, log_prob, entropy, value


def preprocess_obs(obs, device: torch.device) -> torch.Tensor:
    """
    Convert a raw environment observation to a model-ready tensor.

    The environment returns shape (4, 84, 84, 1) due to grayscale_newaxis=True.
    This function squeezes the trailing dim and moves to the correct device.

    Args:
        obs    : numpy array of shape (4, 84, 84, 1) or (4, 84, 84), float32 in [0,1].
        device : torch device.

    Returns:
        Tensor of shape (4, 84, 84) on the given device.
    """
    import numpy as np
    obs = np.array(obs, dtype=np.float32)
    if obs.ndim == 4 and obs.shape[-1] == 1:
        obs = obs.squeeze(-1)   # (4, 84, 84, 1) -> (4, 84, 84)
    return torch.tensor(obs, dtype=torch.float32, device=device)


if __name__ == "__main__":
    """Quick sanity check for the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AtariActorCritic(n_actions=18).to(device)

    # Simulate a batch of 4 observations
    dummy_obs = torch.zeros(4, 4, 84, 84, device=device)
    logits, value = model(dummy_obs)

    print(f"Logits shape : {logits.shape}")   # expected: (4, 18)
    print(f"Value shape  : {value.shape}")    # expected: (4,)

    action, log_prob, entropy, val = model.get_action_and_value(dummy_obs)
    print(f"Action shape   : {action.shape}")    # expected: (4,)
    print(f"Log-prob shape : {log_prob.shape}")  # expected: (4,)
    print(f"Entropy        : {entropy.item():.4f}")
    print(f"Value shape    : {val.shape}")        # expected: (4,)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("model.py OK")