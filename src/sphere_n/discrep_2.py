from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def discrep_2(K: NDArray[Any], X: npt.NDArray[np.float64]) -> float:
    """dispersion measure

    Args:
        K (NDArray[Any]): Array representing indices
        X (NDArray[np.float64]): Array representing points

    Returns:
        float: dispersion
    """
    n = K.shape[1]
    points = X[K]  # (nsimplex, n, ndim)
    iu, ju = np.triu_indices(n, k=1)
    dots = np.einsum("sid,sid->si", points[:, iu], points[:, ju])
    q = 1.0 - dots * dots
    maxq = float(q.max())
    minq = float(q.min())
    dis = np.arcsin(np.sqrt(maxq)) - np.arcsin(np.sqrt(minq))
    return float(dis)
