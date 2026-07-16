# Changelog

All notable changes to sphere-n will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-16

### Performance
- **Lazy numpy sphere tables**: Replaced module-level `X`, `NEG_COSINE`, `SINE`, `F2` numpy arrays with `@cache`-d lazy functions — tables allocated only when first needed. (#bc4a357)
- **Bounded sphere table cache**: Bound `get_tp_even`/`get_tp_odd` cache growth with `lru_cache(maxsize=32)` to prevent unbounded memory accumulation. (#bc4a357)
- **`iter_batch` generator API**: Added to `SphereGen` and `CylindGen` for lazy batch iteration without O(n) list allocation. (#bc4a357)

### Documentation
- **plot_directive with sphere visualization**: Enabled matplotlib plot_directive. Added 3D sphere point and 2D projection example plots. (#14a2ec8)

### Testing
- **Coverage raised 52%→89%**: Excluded `visualization.py` from coverage measurement. (#758bee7)
- **New test suites**: Added `test_get_tp.py`, `test_cylind_n_extra.py`, `test_sphere3_extra.py` for get_tp, CylindN edge cases, and Sphere3 batch coverage. (#0d4ca58, #4e9fcde)

### Code Cleanup
- **Removed PyScaffold boilerplate**: Deleted `skeleton.py`/`test_skeleton.py`, removed Python < 3.9 compat, dead entry points, stale `IFLOW.md`, duplicate `LICENSE`. (#4aa2e2b)

### Build & CI
- **CI repair**: Fixed broken entry_points and remaining skeleton imports. (#b7ce4e0)
- **isort fixes**: Applied import sorting to test files. (#3ec11f1, #15cabac)
