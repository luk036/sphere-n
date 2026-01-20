# Contributing to sphere-n

Thank you for your interest in contributing to sphere-n! This document provides guidelines for contributing effectively.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/luk036/sphere-n.git
cd sphere-n

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .[testing]

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest

# Run linting
flake8 src/sphere_n tests
```

## Code Style

### Formatting

This project uses:
- **Black** for code formatting (256 char line length)
- **isort** for import sorting
- **flake8** for linting

All three are configured in `.pre-commit-config.yaml` and run automatically before commits.

### Type Hints

All functions and methods must have full type annotations:

```python
def my_function(x: int, y: float) -> List[float]:
    """Description of function."""
    return [x * y]
```

### Docstrings

Use Google-style docstrings with Args and Returns sections:

```python
def calculate_mean(data: List[float]) -> float:
    """Calculates the arithmetic mean of data.

    Args:
        data (List[float]): List of numerical values.

    Returns:
        float: The arithmetic mean.
    """
    return sum(data) / len(data)
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `SphereN`, `CylindN`)
- Functions/Methods: `snake_case` (e.g., `get_tp`, `reseed`)
- Variables: `snake_case` (e.g., `maxq`, `minq`)
- Constants: `UPPER_CASE` (e.g., `PI`, `HALF_PI`)

### Error Handling

Use explicit `ValueError` instead of `assert` for input validation:

```python
# GOOD
def __init__(self, n: int) -> None:
    if n < 2:
        raise ValueError(f"Dimension must be >= 2, got {n}")

# AVOID
def __init__(self, n: int) -> None:
    assert n >= 2  # Can be optimized away
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_sphere3.py

# Specific test function
pytest tests/test_sphere3.py::test_sphere3

# With coverage
pytest --cov=sphere_n --cov-report=term-missing

# Without coverage (faster)
pytest --no-cov
```

### Writing Tests

Tests should:
- Use descriptive names (`test_sphere3_dimension`, not `test1`)
- Use `pytest.approx()` for floating-point comparisons
- Be independent (each test should work alone)
- Test both happy paths and edge cases

Example:

```python
def test_sphere_n_normalization() -> None:
    """Test that generated points are normalized."""
    sgen = SphereN([2, 3, 5, 7])
    point = sgen.pop()
    norm = sum(x**2 for x in point) ** 0.5
    assert norm == approx(1.0)
```

### Property-Based Testing

We use Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=10))
def test_dimension_points(n: int) -> None:
    """Test that dimension matches output size."""
    bases = [2, 3, 5, 7][:n]
    sgen = SphereN(bases)
    point = sgen.pop()
    assert len(point) == n + 1
```

## Submitting Changes

### Workflow

1. **Fork** the repository and create a feature branch
2. **Make changes** following code style guidelines
3. **Add tests** for new functionality
4. **Run tests** and linting locally
5. **Commit** changes with descriptive messages
6. **Push** to your fork
7. **Create Pull Request** referencing any related issues

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(sphere_n): add pop_batch method for efficient generation
fix(cylind_n): raise ValueError for invalid dimensions
docs(readme): add quick start section
```

### Pre-commit Checks

Before pushing, ensure all hooks pass:

```bash
# Manually run all pre-commit checks
pre-commit run --all-files

# Or just run specific checks
black src/sphere_n tests
isort src/sphere_n tests
flake8 src/sphere_n tests
```

### Pull Request Guidelines

- Link to related issues using `#123` format
- Keep PRs focused and small (one feature/fix per PR)
- Update documentation for new features
- Add tests for new functionality
- Ensure CI passes (green checkmark)

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS version
- Minimal reproducible code example
- Expected vs actual behavior
- Error traceback (if any)

Template:

```markdown
**Description**: Brief description of the bug

**Reproduction**:
```python
sgen = SphereN([2, 3, 5])
sgen.pop()  # Expected: [...], Actual: [...]
```

**Environment**:
- Python: 3.11
- OS: Ubuntu 22.04
- sphere-n: 1.0.0
```

### Feature Requests

Explain the use case and why it's important. Consider:

- How would you use this feature?
- What problem does it solve?
- Are there workarounds?

### Documentation Issues

- Specify which document (README, API docs, etc.)
- Quote the unclear section
- Suggest clarification if possible

## Development Tools

### IDE Setup

Recommended IDE extensions:
- **Python** (Microsoft): IntelliSense, debugging
- **Pylance**: Type checking, autocomplete
- **Pre-commit**: Visual hook status

### Debugging

For debugging tests:

```bash
# Run with output
pytest -vvs

# Run with breakpoint
pytest --pdb

# Run with coverage visualization
pytest --cov=sphere_n --cov-report=html
# Then open htmlcov/index.html
```

## Getting Help

- Check existing [GitHub issues](https://github.com/luk036/sphere-n/issues)
- Read [AGENTS.md](AGENTS.md) for development guidelines
- Review existing code for patterns

Thank you for contributing! 🎉
