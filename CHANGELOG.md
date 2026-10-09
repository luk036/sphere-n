# Changelog

All notable changes to sphere-n will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-10-09

### Performance
- **Vectorized `discrep_2` dispersion measure**: Replaced the per-pair Python loop with a single pair-indexed einsum over all simplices — ~60× speedup with identical results. (#49954f2)
- **Batch API in dispersion experiments**: Generate point sets via `pop_batch` instead of per-point `pop()` loops. (#8048078)

### Bug Fixes
- **Pandoc-safe paper config**: Split the multi-command `\ifxetex` header-include that pandoc 3.11 mangles and pinned figure floats via `\fps@figure{tb}`. (#a7ed973)
- **mypy config**: Removed the duplicate `ignore_missing_imports` entry. (#1cad724)
- **RTD docs build**: Added `matplotlib` and `numpy` to `docs/requirements.txt`. (#f694c73)

### Testing & Code Quality
- **Coverage raised 89% → 100%**: Added `iter_batch` and invalid-dimension coverage tests. (#1977e02)

### Code Cleanup
- **Consolidated onto `lds_gen.sphere_n`**: Dropped the duplicated numpy `sphere_n.py`; `Sphere3` / `SphereN` / `get_tp` now come from the sibling `lds-gen` package. Rewired visualization, tests, and docs. (#47615a8, #89f28ed, #2ac31da, #e50ea9b)
- **Removed AI slop & fixed import order**: Stripped docstring/comment boilerplate and fixed import order in the experiment and docs scripts. (#51cabe9, #dab19cc, #8eb74d9)

### Documentation
- **n-sphere paper overhaul**: Restructured and expanded the paper (Previous Work, cylindrical-mapping surface-element explanation, lookup-table numerics, recursive-integral indexing fix, dispersion reported as mean ± std over 30 trials), added algorithm pseudocode and a dispersion results table, and polished the prose. (#024e66c, #789c20c, #c602d8a, #6ee2053)
- **Beamer presentation**: Added `n-sphere-talk.md` and the generated 16:9 slides, updated to match the paper. (#181a53f, #5da4649)
- **Paper TODO list**: Added. (#fc45891)

### Build & CI
- **Paper build pipeline**: Added a pdflatex `Makefile` and `svg2pdf.lua` filter with pre-rendered SVG→PDF figure twins, switched the document to IEEEtran with a vendored `ieee.csl`, and loaded the algorithm packages. (#311e55b, #020a3e1, #30cc302, #10aa012)
- **Updated GitHub Actions**: `setup-python`→v5, `codecov-action`→v4; removed the stale `.bak` workflow. (#cbe6431, #3af3d3c)

### Maintenance
- **Stop tracking `refs/`**: Added `/refs/` to `.gitignore` and removed the previously tracked reference PDFs/slides from the index (kept on disk). (#bdbaaf7, #cf95100)
- **Added `.opencode/package-lock.json`**. (#b80c476)

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
