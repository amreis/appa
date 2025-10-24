import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class Net(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class LitNet(L.LightningModule):
    def __init__(
        self,
        model: Net,
        points_to_avoid: T.Tensor,
        barrier_strength: float = 0.5,
        barrier_width: float = 0.02,
        use_log: bool = False,
    ):
        super().__init__()

        self.model = model
        self.points_to_avoid = points_to_avoid
        self.use_log = use_log

        self.barrier_strength = barrier_strength
        self.barrier_width = barrier_width

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters())

    def on_train_start(self):
        self.points_to_avoid = self.points_to_avoid.to(self.device)

    def training_step(self, batch, batch_idx: int):
        x, x_proj = batch

        x_proj_hat = self.model(x + T.randn_like(x) * 0.02)

        loss = F.mse_loss(x_proj_hat, x_proj + T.randn_like(x_proj_hat) * 0.02)

        distances = T.cdist(x_proj_hat, self.points_to_avoid)
        threshold = self.barrier_width
        proximity_to_bad_spots = F.relu(threshold - distances).sum(dim=1).mean()

        # loss = loss + 0.1 * proximity_to_bad_spots
        total_loss = loss + self.barrier_strength * proximity_to_bad_spots
        if self.use_log:
            self.log_dict(
                {"loss": total_loss, "mse_loss": loss, "prox_loss": proximity_to_bad_spots},
                prog_bar=True,
            )

        return total_loss
