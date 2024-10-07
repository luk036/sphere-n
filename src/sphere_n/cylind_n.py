"""
This code defines a system for generating points on an n-dimensional sphere using a technique called cylindrical mapping. The main purpose is for the sake of comparison with the sphere_n.py code.

The code takes a list of integers as input, which are used as bases for the low-discrepancy sequence generators. These generators help create a more uniform distribution of points compared to random sampling.

The output of this code is a list of floating-point numbers representing coordinates on the n-dimensional sphere. Each time you call the pop() method of a CylindN object, it produces a new set of coordinates.

The code achieves its purpose through a recursive algorithm. It uses two main components: a van der Corput sequence generator (VdCorput) for one dimension, and either a Circle generator or another CylindN generator for the remaining dimensions. This recursive structure allows it to handle spheres of any dimension.

The main logic flow happens in the pop() method. It first generates a cosine value (cosphi) using the van der Corput sequence, mapping it to the range [-1, 1]. Then it calculates the sine value (sinphi) using the Pythagorean identity. The method then recursively generates coordinates for lower dimensions and scales them by sinphi, finally adding cosphi as the last coordinate.

An important data transformation occurs in the pop() method, where the uniform distribution from the van der Corput sequence is transformed into a cosine distribution, which is necessary for proper spherical mapping.

The code also includes a reseed() method, which allows you to reset the internal state of the generators. This is useful for reproducibility in scientific simulations or when you need to generate the same sequence of points multiple times.
"""

from abc import abstractmethod, ABC
from typing import List

# import numexpr as ne
from lds_gen.lds import Circle, VdCorput  # low-discrepancy sequence generators
import numpy as np
import math

PI: float = np.pi


class CylindGen(ABC):
    """Base interface for n-sphere generators using cylindrical mapping."""

    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reseeds the generator with a new seed."""
        raise NotImplementedError


class CylindN(CylindGen):
    """Low-discrepency sequence generator using cylindrical mapping.

    Examples:
        >>> cgen = CylindN([2, 3, 5, 7])
        >>> cgen.reseed(0)
        >>> for _ in range(1):
        ...     print(cgen.pop())
        ...
        [0.4702654580212986, 0.5896942325314937, -0.565685424949238, -0.33333333333333337, 0.0]
    """

    def __init__(self, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        n = len(base) - 1
        assert n >= 1
        self.vdc = VdCorput(base[0])
        self.c_gen = Circle(base[1]) if n == 1 else CylindN(base[1:])

    def pop(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        cosphi = 2.0 * self.vdc.pop() - 1.0  # map to [-1, 1]
        sinphi = math.sqrt(1.0 - cosphi * cosphi)
        return [xi * sinphi for xi in self.c_gen.pop()] + [cosphi]

    def reseed(self, seed: int) -> None:
        """_summary_

        Args:
            seed (int): _description_
        """
        self.vdc.reseed(seed)
        self.c_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
