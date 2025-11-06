import numpy as np
import torch as T


@T.no_grad()
def to_float32_tensor(tensor_like: np.ndarray | T.Tensor) -> T.Tensor:
    if not T.is_tensor(tensor_like):
        tensor_like = T.tensor(tensor_like)
    return tensor_like.float()
