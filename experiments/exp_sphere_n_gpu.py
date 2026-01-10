"""
exp_sphere_n_gpu.py

This code is designed to compare how well three different methods can generate
points on a sphere, and then measure how "spread out" those points are.
This version includes a GPU-accelerated dispersion calculation.
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from lds_gen.lds import PRIME_TABLE
from numba import cuda
from rich.progress import track
from scipy.spatial import ConvexHull

from sphere_n.cylind_n import CylindN
from sphere_n.discrep_2 import discrep_2
from sphere_n.sphere_n import SphereN

# import time


def sample_spherical(npoints: int, ndim: int) -> np.ndarray:
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.transpose()


@cuda.jit
def discrep_2_kernel(K, X, max_q_vals, min_q_vals):
    k = cuda.grid(1)  # Thread index, corresponds to simplex index
    if k >= K.shape[0]:
        return

    nsimplex, n = K.shape
    ndim = X.shape[1]

    local_maxq = 0.0
    local_minq = 1.0  # q is 1 - dot^2, so it's between 0 and 1

    for i in range(n - 1):
        for j in range(i + 1, n):
            dot = 0.0
            for dim in range(ndim):
                dot += X[K[k, i], dim] * X[K[k, j], dim]
            q = 1.0 - dot * dot
            if q > local_maxq:
                local_maxq = q
            if q < local_minq:
                local_minq = q

    max_q_vals[k] = local_maxq
    min_q_vals[k] = local_minq


def dispersion_gpu(Triples: np.ndarray) -> float:
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    nsimplex = triangles.shape[0]

    d_X = cuda.to_device(Triples)
    d_K = cuda.to_device(triangles)

    d_max_q_vals = cuda.device_array(nsimplex, dtype=np.float64)
    d_min_q_vals = cuda.device_array(nsimplex, dtype=np.float64)

    threads_per_block = 256
    blocks_per_grid = (nsimplex + (threads_per_block - 1)) // threads_per_block

    discrep_2_kernel[blocks_per_grid, threads_per_block](  # type: ignore
        d_K, d_X, d_max_q_vals, d_min_q_vals
    )

    max_q_vals = d_max_q_vals.copy_to_host()
    min_q_vals = d_min_q_vals.copy_to_host()

    maxq = np.max(max_q_vals)
    minq = np.min(min_q_vals)

    dis = np.arcsin(np.sqrt(maxq)) - np.arcsin(np.sqrt(minq))
    return dis


def dispersion(Triples: np.ndarray) -> float:
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    measure = discrep_2(triangles, Triples)
    return measure


def main() -> None:
    npoints = 2001
    n = 5
    b = PRIME_TABLE[: n - 1]
    Triples_r = sample_spherical(npoints, n)
    spgen = SphereN(b)
    cygen = CylindN(b)
    Triples_s = np.array([spgen.pop() for _ in range(npoints)])
    Triples_c = np.array([cygen.pop() for _ in range(npoints)])

    x = list(range(200, npoints, 100))
    res_r = []
    res_s = []
    res_c = []
    res_s_gpu = []

    use_gpu = cuda.is_available()
    if use_gpu:
        print("CUDA device found, running GPU version for 'Our' method")
    else:
        print("No CUDA device found, running CPU versions only")

    for i in track(x, description="Calculating dispersion"):
        res_r += [dispersion(Triples_r[:i, :])]
        res_c += [dispersion(Triples_c[:i, :])]

        if use_gpu:
            res_s_gpu += [dispersion_gpu(Triples_s[:i, :])]
        else:
            res_s += [dispersion(Triples_s[:i, :])]

    plt.plot(x, res_r, "r", label="Random")
    plt.plot(x, res_c, "b", label="Cylin")
    if use_gpu:
        plt.plot(x, res_s_gpu, "y", label="Our (GPU)")
    else:
        plt.plot(x, res_s, "g", label="Our (CPU)")

    plt.legend(loc="best")
    plt.xlabel("#points")
    plt.ylabel("dispersion")
    plt.show()


if __name__ == "__main__":
    main()
