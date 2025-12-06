from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from lds_gen.lds import Sphere3Hopf
from rich.progress import Progress, track
from scipy.spatial import ConvexHull

from sphere_n.discrep_2 import discrep_2
from sphere_n.sphere_n import Sphere3

# import matplotlib.pylab as lab


def sample_spherical(npoints: int, ndim: int = 3) -> np.ndarray:
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.transpose()


def dispersion(Triples: np.ndarray) -> float:
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    measure = discrep_2(triangles, Triples)
    return measure


def main() -> None:
    npoints = 2001
    ndim = 4
    Triples_r = sample_spherical(npoints, ndim)
    sphopfgen = Sphere3Hopf([2, 3, 5])
    spgen = Sphere3([2, 3, 5])
    Triples_h = np.array([sphopfgen.pop() for _ in range(npoints)])
    Triples_s = np.array([spgen.pop() for _ in range(npoints)])

    x = list(range(100, npoints, 100))
    res_r = []
    res_h = []
    res_s = []

    for i in track(x, description="Calculating dispersion"):
        res_r += [dispersion(Triples_r[:i, :])]
        res_h += [dispersion(Triples_h[:i, :])]
        res_s += [dispersion(Triples_s[:i, :])]

    plt.plot(x, res_r, "r", label="Random")
    plt.plot(x, res_h, "b", label="Hopf")
    plt.plot(x, res_s, "g", label="Our")
    plt.legend(loc="best")
    plt.xlabel("#points")
    plt.ylabel("dispersion")
    plt.show()


if __name__ == "__main__":
    main()
