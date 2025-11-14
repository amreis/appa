import lightning as L
import numpy as np
import torch as T
import torchkde as tkde

from appa.model import APPALitModule, Net
from appa.utils import to_float32_tensor


class APPA:
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        *,
        kde_kernel: str = "gaussian",
        kde_bandwidth: float = 0.01,
        training_epochs: int = 300,
        logging: bool = False,
    ):
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self._logging = logging

        self._grid_size = 300  # factor out 300 into hparam

        self._model = Net(self.input_dim, self.proj_dim)
        self._training_epochs = training_epochs

        self._kde = tkde.KernelDensity(bandwidth=kde_bandwidth, kernel=kde_kernel, eps=1e-5)

    def fit(self, X_high: np.ndarray | T.Tensor, X_proj: np.ndarray | T.Tensor):
        self._model.train()
        X_high = to_float32_tensor(X_high)
        X_proj = to_float32_tensor(X_proj)

        litmodel = APPALitModule(
            self._model,
            self._kde,
            kde_data=X_proj,
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

        return self._model  # maybe return self instead.

    def predict_no_grad(self, inputs: np.ndarray | T.Tensor) -> T.Tensor:
        self._model.eval()
        if not T.is_tensor(inputs):
            inputs = T.tensor(inputs)

        with T.no_grad():
            return self._model(inputs.float())
