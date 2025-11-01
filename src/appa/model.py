import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import logging
import torchkde as tkde

logger = logging.getLogger(__name__)


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


class APPALitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
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

        total_loss = loss + self.barrier_strength * proximity_to_bad_spots
        if self.use_log:
            self.log_dict(
                {"loss": total_loss, "mse_loss": loss, "prox_loss": proximity_to_bad_spots},
                prog_bar=True,
            )

        return total_loss


class DiffAPPALitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        kde_model: tkde.KernelDensity,
        kde_data: T.Tensor,
        use_log=False,
    ):
        super().__init__()

        self.model = model
        self.use_log = use_log
        self._kde = kde_model
        self._kde_data = kde_data

        self.input_noise_std = 0.02
        self.proj_noise_std = 0.02

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters())

    def on_fit_start(self):
        if self.use_log:
            logger.info("setup(): fitting KDE")
        # The data on which KDE is fit must be on the same
        # device as the data we will call score_samples for.
        # Since we only know the device where training will
        # run once fit() is called, we must fit the KDE here,
        # after moving it to the correct device.
        self._kde.fit(self._kde_data.to(self.device))

    def training_step(self, batch, batch_idx):
        x, x_proj = batch

        x_proj_hat = self.model(x + T.randn_like(x) * self.input_noise_std)

        x_proj_perturbed = x_proj_hat + T.randn_like(x_proj_hat) * self.proj_noise_std
        # loss = F.mse_loss(x_proj_hat, x_proj + T.randn_like(x_proj) * self.proj_noise_std)
        loss = F.mse_loss(x_proj_perturbed, x_proj)

        kde_log_prob = self._kde.score_samples(x_proj_perturbed)
        kde_loss = kde_log_prob.clip(max=-2.0).neg().mean()

        loss_total = loss + 0.002 * kde_loss
        if self.use_log:
            self.log_dict({"t_loss": loss_total, "kde_l": kde_loss, "mse_l": loss}, prog_bar=True)

        return loss_total

    def on_train_epoch_start(self):
        if False and self.current_epoch > 350:  # TODO CHANGE
            self.input_noise_std = 0.02
            self.proj_noise_std = 0.02
