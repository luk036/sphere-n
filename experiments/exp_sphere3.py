from __future__ import print_function

import matplotlib.pyplot as plt

# import matplotlib.pylab as lab

from scipy.spatial import ConvexHull
import numpy as np
from sphere_n.sphere_n import Sphere3
from lds_gen.lds import Sphere3Hopf
from sphere_n.discrep_2 import discrep_2


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.transpose()


def dispersion(Triples):
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    measure = discrep_2(triangles, Triples)
    return measure


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

for i in x:
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
