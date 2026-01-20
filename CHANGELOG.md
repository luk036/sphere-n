# Changelog

All notable changes to sphere-n will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `pop_batch()` method for efficient batch point generation
- Added comprehensive edge case testing in `test_edge_cases.py`
- Added performance guide documentation (`docs/PERFORMANCE.md`)
- Added detailed CONTRIBUTING guide for developers

### Changed
- Replaced `assert` statements with proper `ValueError` exceptions for input validation
- Improved docstrings with complete parameter descriptions
- Documented magic constants with inline comments explaining mathematical context
- Updated `SphereGen` abstract base class with batch API support

### Fixed
- Fixed incomplete docstrings in `Sphere3` and `SphereN` classes
- Fixed incomplete docstrings in `CylindN` class

## [1.0.0] - YYYY-MM-DD

### Added
- Initial release with Sphere3, SphereN, and CylindN generators
- Low-discrepancy sequence generation on n-dimensional spheres
- Cylindrical mapping alternative implementation
- Dispersion measure calculation (`discrep_2`)
- Comprehensive test coverage
- Pre-commit hooks configuration
- Documentation with Sphinx

---

## Format Guidelines

### Added
- New features
- New API methods
- New generators

### Changed
- Changes in existing functionality
- Breaking changes (mention migration steps)

### Deprecated
- Features being removed in future versions

### Removed
- Features removed in this version

### Fixed
- Bug fixes
- Regression fixes

### Security
- Security vulnerability fixes
