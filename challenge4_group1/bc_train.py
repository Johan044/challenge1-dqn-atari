import torch


class BCTrainer:

    #Initialization

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

    #Training epoch

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0.0

        for obs_batch, act_batch in dataloader:

            obs_batch = obs_batch.to(self.device)
            act_batch = act_batch.to(self.device)

            logits, _ = self.model(obs_batch)

            loss = self.criterion(
                logits,
                act_batch,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    #Evaluation

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        correct = 0
        total = 0

        for obs_batch, act_batch in dataloader:

            obs_batch = obs_batch.to(self.device)
            act_batch = act_batch.to(self.device)

            logits, _ = self.model(obs_batch)

            preds = logits.argmax(dim=1)

            correct += (preds == act_batch).sum().item()
            total += act_batch.size(0)

        return correct / total