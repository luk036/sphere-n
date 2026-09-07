"""Tests for the get_tp mapping function in lds_gen.sphere_n."""

import math

import numpy as np
from lds_gen.sphere_n import get_tp
from pytest import approx

TABLE_SIZE = 300


def test_get_tp_n0() -> None:
    result = get_tp(0)
    assert len(result) == TABLE_SIZE
    assert result[0] == approx(0.0)
    assert result[-1] == approx(math.pi)


def test_get_tp_n1() -> None:
    result = get_tp(1)
    assert len(result) == TABLE_SIZE
    assert result[0] == approx(-1.0)
    assert result[-1] == approx(1.0)


def test_get_tp_n2() -> None:
    result = get_tp(2)
    assert len(result) == TABLE_SIZE
    assert result[0] == approx(0.0)
    assert result[-1] == approx(math.pi / 2.0)


def test_get_tp_n3() -> None:
    result = get_tp(3)
    assert len(result) == TABLE_SIZE


def test_get_tp_negative_last_odd() -> None:
    """Odd dimensions map to a symmetric interval around 0."""
    result = get_tp(5)
    assert result[0] == approx(-result[-1])


def test_get_tp_values_increasing() -> None:
    """Verify that tp values are monotonically increasing."""
    for n in [2, 3, 4, 5]:
        tp = get_tp(n)
        diffs = np.diff(tp)
        assert np.all(diffs >= 0), f"get_tp({n}) is not monotonically increasing"


def test_get_tp_cache_reuse() -> None:
    """Verify caching works (same result for repeated calls)."""
    r1 = get_tp(4)
    r2 = get_tp(4)
    np.testing.assert_array_almost_equal(r1, r2)
