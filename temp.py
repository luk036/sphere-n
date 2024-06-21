import numpy as np
from lds_gen.lds import Circle, Sphere, VdCorput
from abc import abstractmethod, ABCMeta
from typing import List

PI: float = np.pi
HALF_PI: float = PI / 2.0

X = np.linspace(0.0, PI, 300)
NEG_COSINE = -np.cos(X)
SINE = np.sin(X)


class GLT:
    def __init__(self):
        self.x = X
        self.neg_cosine = NEG_COSINE
        self.sine = SINE


GL = GLT()


class SphereGen(ABCMeta):
    """Base class for sphere generators."""

    @abstractmethod
    def __init__(self, n: int, base: List[int]):
        raise NotImplementedError

    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reseeds the generator with a new seed."""
        raise NotImplementedError

    @abstractmethod
    def get_tp(self) -> np.ndarray:
        """Returns the precomputed values for t_p."""
        raise NotImplementedError


class Sphere3(SphereGen):
    def __init__(self, base: List[int]):
        self.vdc = VdCorput(base[0])
        self.sphere2 = Sphere(base[1:3])
        self.tp = 0.5 * (GL.x - GL.sine * GL.neg_cosine)

    def get_tp(self):
        return self.tp

    def pop(self) -> List[float]:
        """Generates and returns an array of four values."""
        ti = HALF_PI * self.vdc.pop()  # map to [0, pi/2];
        xi = np.interp(ti, GL.x, X)
        cosxi = np.cos(xi)
        sinxi = np.sin(xi)
        s0, s1, s2 = self.sphere2.pop()
        return [sinxi * s0, sinxi * s1, sinxi * s2, cosxi]

    def reseed(self, seed: int) -> None:
        """Reseeds both internal generators."""
        self.vdc.reseed(seed)
        self.sphere2.reseed(seed)


class NSphere(SphereGen):
    def __init__(self, n: int, base: List[int]):
        assert n >= 3, "n must be greater than or equal to 3"
        self.vdc = VdCorput(base[0])
        if n == 3:
            self.s_gen = Sphere3(base[1:4])
            self.tp = GL.neg_cosine
        else:
            self.s_gen = NSphere(n - 1, base[1:])
            self.tp = (
                ((n - 1) * self.s_gen.get_tp()) + GL.neg_cosine * SINE ** (n - 1)
            ) / n

    def get_tp(self):
        return self.tp

    def get_tp_minus1(self) -> np.ndarray:
        """Returns the t_p values of the underlying generator."""
        return self.s_gen.get_tp()

    def pop(self) -> List[float]:
        """Overrides SphereGen method to generate and return values as a vector."""
        vd = self.vdc.pop()
        ti = self.tp[0] + (self.tp[-1] - self.tp[0]) * vd  # map to [t0, tm-1];
        xi = np.interp(ti, self.tp, X)
        sinphi = np.sin(xi)
        res = self.s_gen.pop()
        res = np.array(res) * sinphi
        res = np.append(res, np.cos(xi))
        return res

    def reseed(self, seed: int) -> None:
        """Reseeds both internal generators."""
        self.vdc.reseed(seed)
        self.s_gen.reseed(seed)


class Cylind(ABCMeta):
    """Base interface for cylindrical generators."""

    @abstractmethod
    def __init__(self, n: int, base: List[int]):
        raise NotImplementedError

    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError

    @abstractmethod
    def reseed(self, seed: int) -> None:
        """Reseeds the generator with a new seed."""
        raise NotImplementedError


class CylindN(Cylind):
    """Generate using cylindrical coordinate method."""

    def __init__(self, n: int, base: List[int]):
        assert n >= 1, "n must be greater than or equal to 2"
        self.vdc = VdCorput(base[0])
        self.c_gen: Cylind = Circle(base[1]) if n == 1 else CylindN(n - 1, base[1:])

    def pop(self) -> List[float]:
        """Overrides Cylind method to return cylindrical coordinates as a list."""
        cosphi = 2.0 * self.vdc.pop() - 1.0
        sinphi = np.sqrt(1.0 - cosphi**2)
        return [xi * sinphi for xi in self.c_gen.pop()] + [cosphi]

    def reseed(self, seed: int) -> None:
        """Reseeds both the VdCorput and child generator."""
        self.vdc.reseed(seed)
        self.c_gen.reseed(seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
