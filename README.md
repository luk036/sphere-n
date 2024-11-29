[![codecov](https://codecov.io/gh/luk036/sphere-n/branch/main/graph/badge.svg?token=EIv4D8NlYj)](https://codecov.io/gh/luk036/sphere-n)
[![Documentation Status](https://readthedocs.org/projects/sphere-n/badge/?version=latest)](https://sphere-n.readthedocs.io/en/latest/?badge=latest)

# ⚽ sphere-n

> Generator of Low discrepancy Sequence on S_n

This library implements a generator for the generation of low-discrepancy sequences on n-dimensional spheres. Low-discrepancy sequences are utilized for the generation of points that are distributed uniformly across a given space. This technique is of significant value in a number of fields, including computer graphics, numerical integration, and Monte Carlo simulations.

The principal objective of this library is to facilitate the generation of points on the surface of spheres of varying dimensions, including three-dimensional and higher-dimensional spheres. The input required is the dimension of the sphere (n) and a set of base numbers to be used for the underlying sequence generation. The output is a series of vectors, with each vector representing a point on the surface of the n-dimensional sphere.

The library achieves this through a combination of mathematical calculations and recursive structures. The library utilizes a number of fundamental components, including:

1. The VdCorput sequence generator produces a sequence of numbers that are evenly distributed between 0 and 1.
2. Subsequently, the aforementioned numerical data is mapped onto the surface of a sphere through the use of interpolation functions.
3. The SphereGen module represents an abstract base class that defines the common interface for all sphere generators.
4. The recursive structures, namely Sphere3 and NSphere, facilitate the construction of higher-dimensional spheres from their lower-dimensional counterparts.

The primary logic flow begins with the construction of a SphereN object, which utilizes either a Sphere3 (for 3D) or a recursive process to generate lower-dimensional spheres for higher dimensions. In the generation of points, the VdCorput sequence is employed to obtain a base number, which is then subjected to a series of transformations involving sine, cosine, and interpolation in order to map it onto the surface of the sphere.

A noteworthy aspect of the library is its incorporation of caching (through the @cache decorator) to optimize performance by storing and reusing calculated values. Moreover, the library provides traits and structures that facilitate the adaptable deployment of the sphere generators. The SphereGen abstract base class serves to define a common interface for a variety of sphere generators, whereas the NSphere and SphereN structures are responsible for implementing the actual generation logic.

In conclusion, this library provides a sophisticated yet flexible method for generating evenly distributed points on high-dimensional spheres, which can be advantageous in numerous scientific and computational applications.

## Dependencies

- [luk036/lds-gen](https://github.com/luk036/lds-gen)
- numpy 
- scipy (for testing only)

## 👀 See also

- [sphere-n-cpp](https://github.com/luk036/sphere-n-cpp)
- [sphere-n-rs](https://github.com/luk036/sphere-n-rs)
- [slides](https://luk036.github.io/n_sphere/slides.html)

## 👉 Note

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see <https://pyscaffold.org/>.
