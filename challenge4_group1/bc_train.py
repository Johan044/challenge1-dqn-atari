class BCTrainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0.0

        for obs_batch, act_batch in loader:

            obs_batch = obs_batch.to(self.device)
            act_batch = act_batch.to(self.device)

            logits, _ = self.model(obs_batch)

            loss = self.criterion(
                logits,
                act_batch
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)