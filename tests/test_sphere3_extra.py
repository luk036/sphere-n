"""Additional Sphere3 tests covering edge cases."""

import numpy as np
from pytest import approx

from sphere_n.sphere_n import Sphere3


def test_sphere3_pop_batch() -> None:
    sgen = Sphere3([2, 3, 5])
    sgen.reseed(0)
    batch = sgen.pop_batch(5)
    assert len(batch) == 5
    for vec in batch:
        assert len(vec) == 4
        assert np.linalg.norm(vec) == approx(1.0)


def test_sphere3_reseed_reproducibility() -> None:
    sgen = Sphere3([2, 3, 5])
    sgen.reseed(42)
    first = sgen.pop()
    sgen.reseed(42)
    second = sgen.pop()
    assert first == second


def test_sphere3_normalization() -> None:
    sgen = Sphere3([2, 3, 5])
    for _ in range(10):
        vec = sgen.pop()
        assert np.linalg.norm(vec) == approx(1.0, rel=1e-10)
