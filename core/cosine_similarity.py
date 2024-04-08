from typing import Any

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim


def cosine_similarity(x: Any, y: Any) -> float:

    def _to_numpy(_x):
        if isinstance(_x, torch.Tensor):
            _x = _x.detach().cpu().numpy()
        _x = np.array(_x).reshape((1, -1))
        return _x

    return _cosine_sim(_to_numpy(x), _to_numpy(y)).mean()
