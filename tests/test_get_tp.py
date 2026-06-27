"""Tests for the get_tp mapping functions in sphere_n."""

import numpy as np
from pytest import approx

from sphere_n.sphere_n import get_tp, get_tp_even, get_tp_odd


def test_get_tp_even_n0() -> None:
    result = get_tp_even(0)
    assert len(result) == 300


def test_get_tp_even_n2() -> None:
    result = get_tp_even(2)
    assert len(result) == 300
    assert result[0] == approx(0.0)
    assert result[-1] == approx(np.pi / 2.0)


def test_get_tp_odd_n1() -> None:
    result = get_tp_odd(1)
    assert len(result) == 300
    assert result[0] == approx(-1.0)


def test_get_tp_odd_n3() -> None:
    result = get_tp_odd(3)
    assert len(result) == 300


def test_get_tp_even_via_get_tp() -> None:
    result = get_tp(2)
    expected = get_tp_even(2)
    np.testing.assert_array_almost_equal(result, expected)


def test_get_tp_odd_via_get_tp() -> None:
    result = get_tp(3)
    expected = get_tp_odd(3)
    np.testing.assert_array_almost_equal(result, expected)


def test_get_tp_values_increasing() -> None:
    """Verify that tp values are monotonically increasing."""
    for n in [2, 3, 4, 5]:
        tp = get_tp(n)
        diffs = np.diff(tp)
        assert np.all(diffs >= 0), f"get_tp({n}) is not monotonically increasing"


def test_get_tp_cache_reuse() -> None:
    """Verify caching works (same result for repeated calls)."""
    r1 = get_tp_even(4)
    r2 = get_tp_even(4)
    np.testing.assert_array_almost_equal(r1, r2)
