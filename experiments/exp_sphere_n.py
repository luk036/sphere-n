"""
exp_sphere3.py

This code is designed to compare how well three different methods can generate points on a sphere, and then measure how "spread out" those points are.

The input(s) it takes are:

- The number of points to generate (in this case, 2001)
- The dimension of the space where these points will be generated (in this case, 4)

As for what output(s) it produces, the code generates three sets of points on a sphere using different methods: one set is randomly generated, another set follows a specific pattern called a Hopf generator, and the third set uses a custom method. It then measures how "spread out" each set of points is.

To achieve its purpose, the code first defines two functions to generate these sets of points: sample_spherical generates random points on a sphere, while dispersion calculates how spread out a set of points are. The code then uses these functions to create three sets of points and calculate their dispersion values.

The important logic flow in this code is the way it measures the dispersion of each set of points. Dispersion is like a measure of how "spread out" or "scattered" the points are. The code does this by creating triangles from the points, calculating the area of these triangles, and then averaging them to get an idea of how spread out the points are.

One important data transformation happening in this code is the way it converts the generated points into a format that can be used to calculate their dispersion. This involves converting the points from a 4-dimensional space into a set of coordinates that can be used by the dispersion function.

Overall, exp_sphere3.py is designed to compare how well different methods can generate points on a sphere and measure how spread out those points are.
"""

from __future__ import print_function

import matplotlib.pyplot as plt

# import matplotlib.pylab as lab

from scipy.spatial import ConvexHull
import numpy as np
from sphere_n.sphere_n import SphereN
from sphere_n.cylind_n import CylindN
from sphere_n.discrep_2 import discrep_2
from lds_gen.lds import PRIME_TABLE


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
n = 4
b = PRIME_TABLE[: n - 1]
Triples_r = sample_spherical(npoints, n)
spgen = SphereN(b)
cygen = CylindN(b)
Triples_s = np.array([spgen.pop() for _ in range(npoints)])
Triples_c = np.array([cygen.pop() for _ in range(npoints)])

x = list(range(100, npoints, 100))
res_r = []
res_s = []
res_c = []

for i in x:
    res_r += [dispersion(Triples_r[:i, :])]
    res_s += [dispersion(Triples_s[:i, :])]
    res_c += [dispersion(Triples_c[:i, :])]

plt.plot(x, res_r, "r", label="Random")
plt.plot(x, res_s, "g", label="Our")
plt.plot(x, res_c, "b", label="Cylin")
plt.legend(loc="best")
plt.xlabel("#points")
plt.ylabel("dispersion")
plt.show()
