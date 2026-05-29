class GAILTrainer:

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

        self.bce = torch.nn.BCELoss()