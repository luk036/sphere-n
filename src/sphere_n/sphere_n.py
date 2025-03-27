"""
Sphere N Generator

This code is a Sphere N Generator, which is designed to create points on the surface of spheres in different dimensions. It's a tool that mathematicians, scientists, or computer graphics programmers might use when they need to work with spherical shapes in multiple dimensions.

The main input for this code is a list of integers, which are used as bases for generating sequences of numbers. These bases are used to initialize different types of generators that create points on spheres.

The output of this code is a series of lists containing floating-point numbers. Each list represents a point on the surface of a sphere, with the number of elements in the list corresponding to the dimension of the sphere.

To achieve its purpose, the code uses several mathematical concepts and algorithms. It starts by defining some constants and helper functions that are used in the calculations. These functions (get_tp_odd, get_tp_even, and get_tp) create lookup tables for mapping values in different dimensions.

The code then defines several classes that generate points on spheres:

1. SphereGen: This is an abstract base class that defines the common interface for all sphere generators.

2. Sphere3: This class generates points on a 3-dimensional sphere. It uses a combination of van der Corput sequences and 2-dimensional sphere points to create 3D points.

3. SphereN: This class can generate points on spheres of any dimension (3 or higher). It uses a recursive approach, building higher-dimensional spheres from lower-dimensional ones.

Each of these classes has methods to generate new points (pop) and to reset the generator with a new starting point (reseed).

The code achieves its purpose through a combination of mathematical transformations and recursive algorithms. It uses trigonometric functions (sine, cosine) and interpolation to map values from one range to another. The core idea is to generate sequences of numbers that, when interpreted as coordinates, create an even distribution across the surface of a sphere.

An important aspect of the code is its use of caching (@cache decorator) for some functions. This improves performance by storing the results of expensive calculations so they don't need to be repeated.

Overall, this code provides a flexible way to generate evenly distributed points on spheres of various dimensions, which can be useful in many scientific and graphical applications.
"""

from abc import abstractmethod, ABC
from typing import List

# import numexpr as ne
from lds_gen.lds import Sphere, VdCorput  # low-discrepancy sequence generators
from functools import cache
import numpy as np
import math

PI: float = np.pi
X: np.ndarray = np.linspace(0.0, PI, 300)
NEG_COSINE: np.ndarray = -np.cos(X)
SINE: np.ndarray = np.sin(X)
F2: np.ndarray = (X + NEG_COSINE * SINE) / 2.0
HALF_PI = PI / 2.0


@cache
def get_tp_odd(n: int) -> np.ndarray:
    """table-lookup of mapping function for odd n

    Returns:
        np.ndarray: _description_
    """
    if n == 1:
        return NEG_COSINE
    tp_minus2 = get_tp_odd(n - 2)  # NOQA
    # return ne.evaluate("((n - 1) * tp_minus2 + NEG_COSINE * SINE**(n - 1)) / n")
    return ((n - 1) * tp_minus2 + NEG_COSINE * SINE ** (n - 1)) / n


@cache
def get_tp_even(n: int) -> np.ndarray:
    """table-lookup of mapping function for even n

    Returns:
        np.ndarray: _description_
    """
    if n == 0:
        return X
    tp_minus2 = get_tp_even(n - 2)  # NOQA
    # return ne.evaluate("((n - 1) * tp_minus2 + NEG_COSINE * SINE**(n - 1)) / n")
    return ((n - 1) * tp_minus2 + NEG_COSINE * SINE ** (n - 1)) / n


def get_tp(n: int) -> np.ndarray:
    """table-lookup of mapping function for n

    Returns:
        np.ndarray: _description_
    """
    return get_tp_even(n) if n % 2 == 0 else get_tp_odd(n)


class SphereGen(ABC):
    """Base class for sphere generators."""

    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reseeds the generator with a new seed."""
        raise NotImplementedError


class Sphere3(SphereGen):
    """3-Sphere sequence generator

    Examples:
        >>> sgen = Sphere3([2, 3, 5])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.2913440162992141, 0.8966646826186098, -0.33333333333333337, 6.123233995736766e-17]
    """

    vdc: VdCorput  # van der Corput sequence generator
    sphere2: Sphere  # 2-Sphere generator

    def __init__(self, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        self.vdc = VdCorput(base[0])
        self.sphere2 = Sphere(base[1:3])

    def reseed(self, seed: int) -> None:
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.sphere2.reseed(seed)

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        ti = HALF_PI * self.vdc.pop()  # map to [t0, tm-1]
        xi = np.interp(ti, F2, X)
        cosxi = math.cos(xi)
        sinxi = math.sin(xi)
        return [sinxi * s for s in self.sphere2.pop()] + [cosxi]


class SphereN(SphereGen):
    """Sphere-N sequence generator

    Examples:
        >>> sgen = SphereN([2, 3, 5, 7])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.4809684718990214, 0.6031153874276115, -0.5785601510223212, 0.2649326520763179, 6.123233995736766e-17]
    """

    def __init__(self, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        n = len(base) - 1
        assert n >= 2
        self.vdc = VdCorput(base[0])
        self.s_gen = Sphere(base[1:3]) if n == 2 else SphereN(base[1:])
        self.n = n
        tp = get_tp(n)
        self.range = tp[-1] - tp[0]

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        vd = self.vdc.pop()
        tp = get_tp(self.n)
        ti = tp[0] + self.range * vd  # map to [t0, tm-1]
        xi = np.interp(ti, tp, X)
        sinphi = math.sin(xi)
        return [xi * sinphi for xi in self.s_gen.pop()] + [math.cos(xi)]

    def reseed(self, seed: int) -> None:
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.s_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
