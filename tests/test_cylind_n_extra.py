"""Additional CylindN tests covering edge cases."""

import numpy as np
from pytest import approx

from sphere_n.cylind_n import CylindN


def test_cylind_n_3d() -> None:
    """CylindN([2, 3]) -> n=1 -> 2-sphere -> 3D output."""
    cgen = CylindN([2, 3])
    vec = cgen.pop()
    assert len(vec) == 3
    assert np.linalg.norm(vec) == approx(1.0)


def test_cylind_n_4d() -> None:
    """CylindN([2, 3, 5]) -> n=2 -> 3-sphere -> 4D output."""
    cgen = CylindN([2, 3, 5])
    vec = cgen.pop()
    assert len(vec) == 4
    assert np.linalg.norm(vec) == approx(1.0)


def test_cylind_n_5d() -> None:
    """CylindN([2, 3, 5, 7]) -> n=3 -> 4-sphere -> 5D output."""
    cgen = CylindN([2, 3, 5, 7])
    vec = cgen.pop()
    assert len(vec) == 5


def test_cylind_n_pop_batch() -> None:
    cgen = CylindN([2, 3, 5, 7])
    batch = cgen.pop_batch(3)
    assert len(batch) == 3
    for vec in batch:
        assert len(vec) == 5
        assert np.linalg.norm(vec) == approx(1.0)


def test_cylind_n_reseed_reproducibility() -> None:
    cgen = CylindN([2, 3, 5, 7])
    cgen.reseed(99)
    first = cgen.pop()
    cgen.reseed(99)
    second = cgen.pop()
    assert first == second
