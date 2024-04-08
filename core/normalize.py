from typing import Any

import torch
import numpy as np
from sklearn.preprocessing import normalize as _norm


def normalize(x: Any) -> np.array:
    
    def _to_numpy(_x):
        if isinstance(_x, torch.Tensor):
            _x = _x.detach().cpu().numpy()
        _x = np.array(_x).reshape((1, -1))
        return _x
    
    return _norm(_to_numpy(x)).ravel()
