import numpy as np
import torch

def convert_state_to_onehot(state: np.ndarray) -> torch.Tensor:

    assert state.ndim == 2
    min_value = -3
    num_classes = 12

    state_shifted = state - min_value

    assert np.all((state_shifted >= 0) & (state_shifted < num_classes))

    onehot = np.eye(num_classes, dtype=np.float32)[state_shifted]
    onehot = np.transpose(onehot, (2, 0, 1))

    return torch.from_numpy(onehot)
