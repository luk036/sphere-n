import numpy as np
from typing import Union
import numpy.typing as npt
from numpy.typing import NDArray
from typing import Any


def discrep_2(K: NDArray[Any], X: npt.NDArray[np.float64]) -> float:
    """dispersion measure

    Arguments:
        K (NDArray[Any]): Array representing indices
        X (NDArray[np.float64]): Array representing points

    Returns:
        float: dispersion
    """
    nsimplex, n = K.shape
    maxq = 0
    minq = np.inf
    for k in range(nsimplex):
        p = X[K[k, :], :]
        for i in range(n - 1):
            for j in range(i + 1, n):
                dot = np.dot(p[i, :], p[j, :])
                q = 1.0 - dot * dot
                maxq = max(maxq, q)
                minq = min(minq, q)
    dis = np.arcsin(np.sqrt(maxq)) - np.arcsin(np.sqrt(minq))
    return dis
