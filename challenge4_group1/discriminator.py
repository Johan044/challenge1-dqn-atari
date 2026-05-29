import torch
import torch.nn as nn


class GAILDiscriminator(nn.Module):
    """
    Discriminator D(s) -> P(expert | s)

    Obs-only version (más simple y suficiente para Montezuma).
    """

    def __init__(self, n_actions: int = None):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.Tanh(),

            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs):
        feats = self.cnn(obs)
        return self.fc(feats).squeeze(-1)