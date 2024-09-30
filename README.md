[![codecov](https://codecov.io/gh/luk036/sphere-n/branch/main/graph/badge.svg?token=EIv4D8NlYj)](https://codecov.io/gh/luk036/sphere-n)
[![Documentation Status](https://readthedocs.org/projects/sphere-n/badge/?version=latest)](https://sphere-n.readthedocs.io/en/latest/?badge=latest)

# 🏐 sphere-n

> Generator of Low discrepancy Sequence on S_n

This code implements a generator for creating low-discrepancy sequences on n-dimensional spheres. Low-discrepancy sequences are used to generate points that are evenly distributed across a space, which is useful in various fields like computer graphics, numerical integration, and Monte Carlo simulations.

The main purpose of this code is to provide a way to generate points on the surface of spheres of different dimensions (3D and higher). It takes as input the dimension of the sphere (n) and a set of base numbers used for the underlying sequence generation. The output is a series of vectors, where each vector represents a point on the surface of the n-dimensional sphere.

The code achieves this through a combination of mathematical calculations and recursive structures. It uses several key components:

1. The VdCorput sequence generator, which produces evenly distributed numbers between 0 and 1.
2. Interpolation functions to map these numbers onto the surface of a sphere.
3. SphereGen: This is an abstract base class that defines the common interface for all sphere generators.
4. Recursive structures (Sphere3 and NSphere) to build up from lower dimensions to higher ones.

The main logic flow starts with the creation of a SphereN object, which internally uses either a Sphere3 (for 3D) or recursively creates lower-dimensional spheres for higher dimensions. When generating points, it uses the VdCorput sequence to get a base number, then applies various transformations involving sine, cosine, and interpolation to map this onto the sphere's surface.

An important aspect of the code is its use of caching (with the @cache decorater) to improve performance by storing and reusing calculated values.

The code also provides traits and structures to allow for flexible use of the sphere generators. The SphereGen trait defines a common interface for different sphere generators, while the NSphere and SphereN structures implement the actual generation logic.

Overall, this code provides a sophisticated yet flexible way to generate evenly distributed points on high-dimensional spheres, which can be valuable in many scientific and computational applications.


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
