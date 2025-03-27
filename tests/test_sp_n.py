"""
test_sp_n.py

This code is a test suite for evaluating different methods of generating points on the surface of a high-dimensional sphere. It includes functions for generating random points, calculating a dispersion measure, and running tests on different point generation methods.

The main purpose of this code is to compare the quality of point distributions on a sphere using both random generation and low-discrepancy sequences (LDS). It does this by generating a set of points, creating a convex hull from those points, and then calculating a dispersion measure based on the triangles formed by the hull.

The code doesn't take any direct inputs from the user. Instead, it uses predefined parameters like the number of points to generate (600) and the dimensions of the sphere (5D in the random case, 4D in the LDS cases).

The primary outputs are the dispersion measures calculated for each method. These measures are then compared against expected values in the test functions.

The code achieves its purpose through several steps:

1. It defines a function discrep_2 that calculates a dispersion measure for a set of points. This measure is based on the minimum and maximum angles between pairs of points in each simplex (triangle in higher dimensions) of the convex hull.

2. It includes a function random_point_on_sphere that generates a random point on the surface of a sphere in any number of dimensions.

3. The run_random function generates 600 random points on a 5D sphere, creates a convex hull, and calculates the dispersion measure.

4. The run_lds function does the same, but uses a provided generator (either SphereN or CylindN) to create the points instead of random generation.

5. Finally, there are three test functions that run these methods and compare the results to expected values:
    - test_random checks the random point generation
    - test_sphere_n checks the SphereN generator
    - test_cylin_n checks the CylindN generator

The key logic flow involves generating points, creating a convex hull, and then calculating the dispersion measure. The dispersion measure itself involves finding the minimum and maximum angles between pairs of points in each simplex of the hull.

This code is important because it allows comparison between random and deterministic (LDS) methods of generating points on a sphere, which can be crucial in various scientific and mathematical applications where uniform distribution of points is needed.
"""

import numpy as np
from pytest import approx
from scipy.spatial import ConvexHull

from sphere_n.sphere_n import SphereN
from sphere_n.cylind_n import CylindN
from sphere_n.discrep_2 import discrep_2


# Write a function that returns a random point on the surface of a sphere
# in n dimensions
def random_point_on_sphere(n):
    # Generate a random point on the surface of a sphere in n dimensions
    # by generating a random vector and normalizing it
    x = np.random.randn(n)
    x /= np.linalg.norm(x)
    return x


def run_random():
    # reseed
    np.random.seed(1234)
    npoints = 600
    Triples = np.array([random_point_on_sphere(5) for _ in range(npoints)])
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    return discrep_2(triangles, Triples)


def run_lds(spgen):
    npoints = 600
    Triples = np.array([spgen.pop() for _ in range(npoints)])
    hull = ConvexHull(Triples)
    triangles = hull.simplices
    return discrep_2(triangles, Triples)


def test_random():
    measure = run_random()
    assert measure == approx(1.115508637826039)


def test_sphere_n():
    spgen = SphereN([2, 3, 5, 7])
    spgen.reseed(0)
    measure = run_lds(spgen)
    assert measure == approx(0.9125914)


def test_cylin_n():
    cygen = CylindN([2, 3, 5, 7])
    cygen.reseed(0)
    measure = run_lds(cygen)
    assert measure == approx(1.0505837105828988)
