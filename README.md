[![codecov](https://codecov.io/gh/luk036/sphere-n/branch/main/graph/badge.svg?token=EIv4D8NlYj)](https://codecov.io/gh/luk036/sphere-n)
[![Documentation Status](https://readthedocs.org/projects/sphere-n/badge/?version=latest)](https://sphere-n.readthedocs.io/en/latest/?badge=latest)

# 🏐 sphere-n

> Generator of Low discrepancy Sequence on S_n

This Python code is a Sphere-N Generator, which is designed to create points on the surface of spheres in different dimensions. It's a tool that mathematicians, scientists, or computer graphics programmers might use when they need to work with spherical shapes in multiple dimensions.

To achieve its purpose, the code uses several mathematical concepts and algorithms. It starts by defining some constants and helper functions that are used in the calculations. These functions (get_tp_odd, get_tp_even, and get_tp) create lookup tables for mapping values in different dimensions.

The code then defines several classes that generate points on spheres:

1. SphereGen: This is an abstract base class that defines the common interface for all sphere generators.

2. Sphere3: This class generates points on a 3-dimensional sphere. It uses a combination of Van der Corput sequences and 2-dimensional sphere points to create 3D points.

3. SphereN: This class can generate points on spheres of any dimension (3 or higher). It uses a recursive approach, building higher-dimensional spheres from lower-dimensional ones.

4. CylinN: This class generates points on spheres using a cylindrical mapping approach. It's for the sake of comparison with SphereN.

Each of these classes has methods to generate new points (pop) and to reset the generator with a new starting point (reseed).

The code achieves its purpose through a combination of mathematical transformations and recursive algorithms. It uses trigonometric functions (sine, cosine) and interpolation to map values from one range to another. The core idea is to generate sequences of numbers that, when interpreted as coordinates, create an even distribution across the surface of a sphere.

## Dependencies

- [luk036/lds-gen](https://github.com/luk036/lds-gen)
- numpy 
- scipy (for testing only)

## 👀 See also

- [sphere-n-cpp](https://github.com/luk036/sphere-n-cpp)
- [sphere-n-rs](https://github.com/luk036/sphere-n-rs)

## 👉 Note

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see <https://pyscaffold.org/>.
