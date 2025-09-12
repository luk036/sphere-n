"""Generates points on n-dimensional spheres using cylindrical mapping."""

import math
from abc import ABC, abstractmethod
from typing import List

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
        """Reseeds the generator with a new seed."""
        raise NotImplementedError


class CylindN(CylindGen):
    """Low-discrepancy sequence generator using cylindrical mapping.

    Examples:
        >>> cgen = CylindN([2, 3, 5, 7])
        >>> cgen.reseed(0)
        >>> for _ in range(1):
        ...     print(cgen.pop())
        ...
        [0.4702654580212986, 0.5896942325314937, -0.565685424949238, -0.33333333333333337, 0.0]
    """

    def __init__(self, base: List[int]) -> None:
        """Initializes the n-cylinder generator.

        Args:
            base (List[int]): The base for the van der Corput sequence.
        """
        n = len(base) - 1
        assert n >= 1
        self.vdc = VdCorput(base[0])
        self.c_gen = Circle(base[1]) if n == 1 else CylindN(base[1:])

    def pop(self) -> List[float]:
        """Generates a new point on the n-cylinder.

        Returns:
            List[float]: A new point on the n-cylinder.
        """
        cosphi = 2.0 * self.vdc.pop() - 1.0  # map to [-1, 1]
        sinphi = math.sqrt(1.0 - cosphi * cosphi)
        return [xi * sinphi for xi in self.c_gen.pop()] + [cosphi]

    def reseed(self, seed: int) -> None:
        """Reseeds the generator.

        Args:
            seed (int): The new seed.
        """
        self.vdc.reseed(seed)
        self.c_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
