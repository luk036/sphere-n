"""Additional tests to improve sphere_n code coverage."""

from pytest import approx, raises

from sphere_n.cylind_n import CylindN
from lds_gen.sphere_n import SphereN


class TestCylindN:
    """Tests covering CylindN edge cases."""

    def test_cylind_n_invalid_dim(self) -> None:
        """Test CylindN rejects dimension < 1."""
        with raises(ValueError, match="Dimension n must be >= 1"):
            CylindN([2])

    def test_cylind_n_reseed(self) -> None:
        """Test CylindN reseed produces consistent results."""
        cgen = CylindN([2, 3, 5])
        cgen.reseed(42)
        first = cgen.pop()
        cgen.reseed(42)
        second = cgen.pop()
        assert first == second

    def test_cylind_n_pop_batch(self) -> None:
        """Test CylindN pop_batch returns correct count."""
        cgen = CylindN([2, 3, 5, 7])
        batch = cgen.pop_batch(5)
        assert len(batch) == 5
        assert all(len(p) > 0 for p in batch)

    def test_cylind_n_iter_batch(self) -> None:
        """Test CylindN iter_batch yields the correct number of points."""
        cgen = CylindN([2, 3, 5, 7])
        batch = list(cgen.iter_batch(3))
        assert len(batch) == 3
        assert all(len(p) > 0 for p in batch)

    def test_cylind_n_iter_batch_invalid(self) -> None:
        """Test CylindN iter_batch rejects non-positive counts."""
        cgen = CylindN([2, 3, 5, 7])
        with raises(ValueError, match="n must be positive"):
            list(cgen.iter_batch(0))


class TestSphereN:
    """Tests covering SphereN edge cases."""

    def test_sphere_n_reseed(self) -> None:
        """Test SphereN reseed produces consistent results."""
        sgen = SphereN([2, 3, 5])
        sgen.reseed(42)
        first = sgen.pop()
        sgen.reseed(42)
        second = sgen.pop()
        assert first == second

    def test_sphere_n_pop_batch(self) -> None:
        """Test SphereN pop_batch returns correct count."""
        sgen = SphereN([2, 3, 5, 7])
        batch = sgen.pop_batch(3)
        assert len(batch) == 3
        assert all(len(p) > 0 for p in batch)

    def test_sphere_n_values(self) -> None:
        """Test SphereN produces expected first point."""
        sgen = SphereN([2, 3, 5])
        sgen.reseed(0)
        result = sgen.pop()
        assert result[0] == approx(0.2913440162992141)

    def test_sphere_n_invalid_dim(self) -> None:
        """Test SphereN rejects dimension < 2."""
        with raises(AssertionError):
            SphereN([2, 3])

    def test_sphere_n_iter_batch(self) -> None:
        """Test SphereN iter_batch yields the correct number of points."""
        sgen = SphereN([2, 3, 5, 7])
        batch = list(sgen.iter_batch(3))
        assert len(batch) == 3
        assert all(len(p) > 0 for p in batch)

    def test_sphere_n_iter_batch_invalid(self) -> None:
        """Test SphereN iter_batch rejects non-positive counts."""
        sgen = SphereN([2, 3, 5, 7])
        with raises(ValueError, match="n must be positive"):
            list(sgen.iter_batch(0))
