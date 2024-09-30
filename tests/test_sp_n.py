"""
tests\test_sp_n.py

This code is a test file that checks the functionality of two different geometric generators: SphereN and CylinN. The main purpose of this code is to ensure that these generators produce points in a specific geometric space with expected properties.

The code doesn't take any direct inputs from the user. Instead, it uses predefined parameters to create instances of SphereN and CylinN generators. The output of this code is not directly visible to the user, but it produces test results that indicate whether the generators are working correctly or not.

The file contains two main test functions: test_sphere_n() and test_cylin_n(). Each of these functions creates a generator (SphereN or CylinN), runs a low-discrepancy sequence (LDS) test on it, and then checks if the result matches an expected value.

The core of the testing process is the run_lds() function. This function generates 600 points using the provided generator, creates a convex hull from these points, and then calculates a dispersion measure using the discrep_2() function. The dispersion measure is a way to quantify how well-distributed the points are in the geometric space.

The discrep_2() function is the heart of the dispersion calculation. It takes a set of simplices (triangles in this case) and a set of points, and calculates the maximum and minimum angles between pairs of points. The difference between the arcsine of the square root of these angles gives the dispersion measure.

The test functions then compare the calculated dispersion measure to an expected value using the approx() function, which allows for small differences due to floating-point arithmetic.

The code uses several important concepts from geometry and linear algebra, such as convex hulls, dot products, and trigonometric functions. However, the main flow of the code is straightforward: generate points, calculate their distribution properties, and compare the result to an expected value.

This testing approach helps ensure that the SphereN and CylinN generators are producing points with the expected distribution properties, which is crucial for their intended use in whatever larger system they're a part of.
"""
import numpy as np
from pytest import approx
from scipy.spatial import ConvexHull

from sphere_n.discrep_2 import discrep_2
from sphere_n.sphere_n import CylinN, SphereN


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
    measure = run_lds(spgen)
    assert measure == approx(0.9125914)
    # assert measure < 0.913
    # assert measure > 0.912


def test_cylin_n():
    cygen = CylinN([2, 3, 5, 7])
    measure = run_lds(cygen)
    assert measure == approx(1.0505837105828988)
    # assert measure < 1.086
    # assert measure > 1.085
