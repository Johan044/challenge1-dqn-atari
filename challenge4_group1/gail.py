import torch
import torch.nn as nn


class GAILTrainer:

    #Initialization

    def __init__(
        self,
        policy,
        discriminator,
        policy_optimizer,
        disc_optimizer,
        device,
    ):
        self.policy = policy
        self.discriminator = discriminator

        self.policy_optimizer = policy_optimizer
        self.disc_optimizer = disc_optimizer

        self.device = device

        self.bce = nn.BCELoss()

    #Discriminator update

    def update_discriminator(
        self,
        expert_obs,
        agent_obs,
    ):

        self.discriminator.train()

        d_expert = self.discriminator(expert_obs)

        d_agent = self.discriminator(agent_obs)

        expert_loss = self.bce(
            d_expert,
            torch.ones_like(d_expert),
        )

        agent_loss = self.bce(
            d_agent,
            torch.zeros_like(d_agent),
        )

        loss = expert_loss + agent_loss

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        expert_acc = (d_expert > 0.5).float().mean()

        agent_acc = (d_agent < 0.5).float().mean()

        disc_acc = (expert_acc + agent_acc) / 2.0

        return {
            "loss": loss.item(),
            "accuracy": disc_acc.item(),
            "expert_acc": expert_acc.item(),
            "agent_acc": agent_acc.item(),
        }

    #Adversarial reward

    @torch.no_grad()
    def adversarial_reward(
        self,
        obs_batch,
    ):

        self.discriminator.eval()

        d_scores = self.discriminator(obs_batch)

        rewards = torch.log(
            d_scores + 1e-8
        )

        return rewards