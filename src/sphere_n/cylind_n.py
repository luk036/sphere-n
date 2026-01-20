r"""Generates points on n-dimensional spheres using cylindrical mapping.

.. svgbob::
   :align: center

   Cylindrical Mapping:

   +------------------+
   |      ^           |
   |      | z         |
   |      |           |
   |   ___|___        |
   |  /       \       |
   | |         |      |----> Sphere(n)
   |  \_______/       |
   |      |           |
   |      |           |
   |      v           |
   +------------------+

Algorithm:

.. svgbob::
   :align: center

           VdCorput Sequence
                  |
                  v
   [-1,1] <-----------------> Sphere(n)
       cosφ    Cylindrical
              Mapping &
              Normalization

"""

import math
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from lds_gen.lds import Circle, VdCorput  # low-discrepancy sequence generators

PI: float = np.pi


class CylindGen(ABC):
    """Base interface for n-sphere generators using cylindrical mapping."""

    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reseeds the generator.

        Args:
            seed (int): The new seed.
        """

    def pop_batch(self, n: int) -> List[List[float]]:
        """Generates and returns n points in batch.

        This is more efficient than calling pop() n times due to reduced
        Python loop overhead.

        Args:
            n (int): Number of points to generate.

        Returns:
            List[List[float]]: List of n points, each as a vector.

        Examples:
            >>> cgen = CylindN([2, 3, 5, 7])
            >>> cgen.reseed(0)
            >>> cgen.pop_batch(3)
            [[...], [...], [...]]
        """
        return [self.pop() for _ in range(n)]


class CylindN(CylindGen):
    """Low-discrepancy sequence generator using cylindrical mapping.

    Examples:
        >>> cgen = CylindN([2, 3, 5, 7])
        >>> cgen.reseed(0)
        >>> for _ in range(1):
        ...     print(cgen.pop())
        ...
        [0.4702654580212986, 0.5896942325314937, -0.565685424949238, 0.33333333333333337, 0.0]
    """

    def __init__(self, base: List[int]) -> None:
        """Initializes the n-cylinder generator.

        Args:
            base (List[int]): List of integers representing bases for van der Corput
                           sequences at each dimension level. Length must be at least 2.
        """
        n = len(base) - 1
        if n < 1:
            raise ValueError(f"Dimension n must be >= 1, got {n}")
        self.vdc: VdCorput = VdCorput(base[0])
        self.c_gen: Union[Circle, CylindN] = (
            Circle(base[1]) if n == 1 else CylindN(base[1:])
        )

    def pop(self) -> List[float]:
        """Generates and returns a new point on the n-cylinder.

        Returns:
            List[float]: A new point on the n-cylinder.
        """
        cosphi: float = 2.0 * self.vdc.pop() - 1.0  # map to [-1, 1]
        sinphi: float = math.sqrt(1.0 - cosphi * cosphi)
        return [xi * sinphi for xi in self.c_gen.pop()] + [cosphi]

    def reseed(self, seed: int) -> None:
        """Reseeds the generator.

        Args:
            seed (int): The new seed.
        """
        self.vdc.reseed(seed)
        self.c_gen.reseed(seed)

    def pop_batch(self, n: int) -> List[List[float]]:
        """Generates and returns n points in batch.

        This is more efficient than calling pop() n times due to reduced
        Python loop overhead.

        Args:
            n (int): Number of points to generate.

        Returns:
            List[List[float]]: List of n points, each as a vector.

        Examples:
            >>> cgen = CylindN([2, 3, 5, 7])
            >>> cgen.reseed(0)
            >>> cgen.pop_batch(3)
            [[...], [...], [...]]
        """
        return [self.pop() for _ in range(n)]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
