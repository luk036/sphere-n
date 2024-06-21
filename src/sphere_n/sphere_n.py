from functools import lru_cache
import math
from abc import abstractmethod, ABC
from typing import List

# import numexpr as ne
import numpy as np
from lds_gen.lds import Circle, Sphere, VdCorput

PI: float = np.pi
HALF_PI: float = PI / 2.0
X: np.ndarray = np.linspace(0.0, PI, 300)
NEG_COSINE: np.ndarray = -np.cos(X)
SINE: np.ndarray = np.sin(X)


@lru_cache
def get_tp(n: int) -> np.ndarray:
    """_summary_

    Returns:
        np.ndarray: _description_
    """
    if n == 0:
        return X
    if n == 1:
        return NEG_COSINE
    tp_minus2 = get_tp(n - 2)  # NOQA
    # return ne.evaluate("((n - 1) * tp_minus2 + NEG_COSINE * SINE**(n - 1)) / n")
    return ((n - 1) * tp_minus2 + NEG_COSINE * SINE ** (n - 1)) / n


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
    """Sphere3 sequence generator

    Examples:
        >>> sgen = Sphere3([2, 3, 5])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.8966646826186098, 0.2913440162992141, -0.33333333333333337, 6.123233995736766e-17]
    """

    vdc: VdCorput
    sphere2: Sphere

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
        ti = HALF_PI * self.vdc.pop()  # map to [0, pi/2]
        xi = np.interp(ti, get_tp(2), X)
        cosxi = math.cos(xi)
        sinxi = math.sin(xi)
        return [sinxi * s for s in self.sphere2.pop()] + [cosxi]


class SphereN(SphereGen):
    """SphereN sequence generator

    Examples:
        >>> sgen = SphereN(3, [2, 3, 5, 7])
        >>> sgen.reseed(0)
        >>> for _ in range(1):
        ...     print(sgen.pop())
        ...
        [0.6031153874276115, 0.4809684718990214, -0.5785601510223212, 0.2649326520763179, 6.123233995736766e-17]
    """

    def __init__(self, n: int, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        assert n >= 2
        self.vdc = VdCorput(base[0])
        self.s_gen = Sphere(base[1:3]) if n == 2 else SphereN(n - 1, base[1:])
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


class Cylind(ABC):
    """Base interface for cylindrical generators."""

    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reseeds the generator with a new seed."""
        raise NotImplementedError


class CylinN(Cylind):
    """CylinN sequence generator

    Examples:
        >>> cgen = CylinN(3, [2, 3, 5, 7])
        >>> cgen.reseed(0)
        >>> for _ in range(1):
        ...     print(cgen.pop())
        ...
        [0.5896942325314937, 0.4702654580212986, -0.565685424949238, -0.33333333333333337, 0.0]
    """

    def __init__(self, n: int, base: List[int]) -> None:
        """_summary_

        Args:
            base (List[int]): _description_
        """
        assert n >= 1
        self.vdc = VdCorput(base[0])
        self.c_gen = Circle(base[1]) if n == 1 else CylinN(n - 1, base[1:])

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
