"""Generates points on n-dimensional spheres."""

import math
from abc import ABC, abstractmethod
from functools import cache
from typing import List

import numpy as np
from lds_gen.lds import Sphere, VdCorput  # low-discrepancy sequence generators

PI: float = np.pi
X: np.ndarray = np.linspace(0.0, PI, 300)
NEG_COSINE: np.ndarray = -np.cos(X)
SINE: np.ndarray = np.sin(X)
F2: np.ndarray = (X + NEG_COSINE * SINE) / 2.0
HALF_PI = PI / 2.0


@cache
def get_tp_recursive(n: int) -> np.ndarray:
    """Recursively calculates the table-lookup of the mapping function for n.

    Args:
        n (int): The dimension.

    Returns:
        np.ndarray: The table-lookup of the mapping function.
    """
    if n == 0:
        return X
    if n == 1:
        return NEG_COSINE
    tp_minus2 = get_tp_recursive(n - 2)
    return ((n - 1) * tp_minus2 + NEG_COSINE * SINE ** (n - 1)) / n


def get_tp(n: int) -> np.ndarray:
    """Calculates the table-lookup of the mapping function for n.

    Args:
        n (int): The dimension.

    Returns:
        np.ndarray: The table-lookup of the mapping function.
    """
    return get_tp_recursive(n)


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
    """Sphere-N sequence generator.

    Examples:
        >>> sgen = SphereN([2, 3, 5, 7])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.4809684718990214, 0.6031153874276115, -0.5785601510223212, 0.2649326520763179, 6.123233995736766e-17]
    """

    def __init__(self, base: List[int]) -> None:
        """Initializes the n-sphere generator.

        Args:
            base (List[int]): The base for the van der Corput sequence.
        """
        n = len(base) - 1
        assert n >= 2
        self.vdc = VdCorput(base[0])
        if n == 2:
            self.s_gen = Sphere(base[1:3])
        else:
            self.s_gen = SphereN(base[1:])
        self.n = n
        tp = get_tp(n)
        self.range = tp[-1] - tp[0]

    def pop(self) -> List[float]:
        """Generates a new point on the n-sphere.

        Returns:
            List[float]: A new point on the n-sphere.
        """
        if self.n == 2:
            ti = HALF_PI * self.vdc.pop()  # map to [t0, tm-1]
            xi = np.interp(ti, F2, X)
            cosxi = math.cos(xi)
            sinxi = math.sin(xi)
            return [sinxi * s for s in self.s_gen.pop()] + [cosxi]

        vd = self.vdc.pop()
        tp = get_tp(self.n)
        ti = tp[0] + self.range * vd  # map to [t0, tm-1]
        xi = np.interp(ti, tp, X)
        sinphi = math.sin(xi)
        return [xi * sinphi for xi in self.s_gen.pop()] + [math.cos(xi)]

    def reseed(self, seed: int) -> None:
        """Reseeds the generator.

        Args:
            seed (int): The new seed.
        """
        self.vdc.reseed(seed)
        self.s_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
