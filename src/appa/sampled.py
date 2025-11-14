from typing import overload

import lightning as L
import numpy as np
import torch as T
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import minmax_scale

from appa.model import Net, SAPPALitModule
from appa.utils import to_float32_tensor


class APPA:
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        *,
        barrier_strength: float = 0.5,
        barrier_width: float = 0.02,
        training_epochs: int = 300,
        logging: bool = False,
    ):
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self._logging = logging

        self.barrier_strength = barrier_strength
        self.barrier_width = barrier_width

        self._grid_size = 300  # factor out 300 into hparam

        self._model = Net(self.input_dim, self.proj_dim)
        self._training_epochs = training_epochs

    def fit(self, X_high: np.ndarray | T.Tensor, X_proj: np.ndarray | T.Tensor):
        self._model.train()
        X_high = to_float32_tensor(X_high)
        X_proj = to_float32_tensor(X_proj)

        sampled_barrier = self._compute_barrier_function(X_proj)

        litmodel = SAPPALitModule(
            self._model,
            sampled_barrier,
            barrier_strength=self.barrier_strength,
            barrier_width=self.barrier_width,
            use_log=self._logging,
        )

        trainer = L.Trainer(
            max_epochs=self._training_epochs,
            log_every_n_steps=1 if self._logging else None,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=self._logging,
            logger=self._logging,
        )

        train_ds = T.utils.data.TensorDataset(X_high, X_proj)
        train_dl = T.utils.data.DataLoader(train_ds, batch_size=1024, shuffle=True)

        trainer.fit(model=litmodel, train_dataloaders=train_dl)

        return self._model

    def predict_no_grad(self, inputs: np.ndarray | T.Tensor) -> T.Tensor:
        self._model.eval()
        if not T.is_tensor(inputs):
            inputs = T.tensor(inputs)

        with T.no_grad():
            return self._model(inputs)

    @overload
    def _compute_barrier_function(self, X_proj: T.Tensor) -> T.Tensor: ...
    @overload
    def _compute_barrier_function(
        self, X_proj: T.Tensor, *, return_prob: bool = False
    ) -> T.Tensor | tuple[T.Tensor, np.ndarray]: ...
    def _compute_barrier_function(
        self, X_proj: T.Tensor, *, return_prob: bool = False
    ) -> T.Tensor | tuple[T.Tensor, np.ndarray]:
        grid = self._make_2d_grid(self._grid_size)

        kde = KernelDensity(kernel="gaussian", bandwidth=0.01)
        kde.fit(X_proj.numpy())
        Z = kde.score_samples(grid)

        neighbors = NearestNeighbors(n_neighbors=1, p=2).fit(X_proj.numpy())
        D = neighbors.kneighbors(grid)[0].squeeze()

        sample_prob_2d = minmax_scale(1 / (1000 + np.exp(Z))) / (1 + D)
        sample_prob_2d = sample_prob_2d.reshape((self._grid_size, self._grid_size))

        rnd = np.random.random_sample(size=(10000, 2))
        bin_width = 1.0 / self._grid_size
        bin_ix = np.clip((rnd / bin_width).round().astype(int), 0, self._grid_size - 1)
        keep = sample_prob_2d[bin_ix[:, 1], bin_ix[:, 0]] >= np.quantile(sample_prob_2d, 0.8)

        sampled_barrier = T.from_numpy(rnd[keep].copy()).float()
        if return_prob:
            return sampled_barrier, sample_prob_2d
        return sampled_barrier

    def _make_2d_grid(self, grid_size: int) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, grid_size)
        ys = np.linspace(0.0, 1.0, grid_size)
        xx, yy = np.meshgrid(xs, ys)
        xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        return xy
