# AGENTS.md - Guidelines for AI Agents Working on sphere-n

## Build, Lint, and Test Commands

### Building
```bash
pip install -e .[testing]
python setup.py sdist bdist_wheel
```

### Testing
```bash
pytest                              # Run all tests with coverage
pytest tests/test_sphere3.py         # Run specific test file
pytest tests/test_sphere3.py::test_sphere3  # Run specific test
pytest --no-cov -v                  # Without coverage (faster)
pytest -vv                          # Verbose output
```

### Linting and Formatting
```bash
black src/sphere_n tests             # Format code
isort src/sphere_n tests            # Sort imports
flake8 src/sphere_n tests           # Lint
pre-commit run --all-files          # Run all hooks
```

### Documentation
```bash
cd docs && make html                # Build docs
```

## Code Style Guidelines

### Import Order (PEP 8, enforced by isort)
```python
# 1. Standard library
import math
from abc import ABC, abstractmethod
from typing import List, Union

# 2. Third-party
import numpy as np
from lds_gen.lds import Circle, VdCorput

# 3. Local
from sphere_n.sphere_n import SphereGen
```

### Type Annotations
- Required: All functions must have full type hints, including `-> None` for void
- Use `List`, `Union`, `NDArray` from typing/numpy.typing
- No type suppression (`as any`, `@ts-ignore`, etc.)

```python
def get_tp(n: int) -> np.ndarray:
    """Calculates the table-lookup of the mapping function for n."""
    return get_tp_recursive(n)

class SphereGen(ABC):
    @abstractmethod
    def pop(self) -> List[float]:
        """Generates and returns a vector of values."""
        raise NotImplementedError
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `SphereN`, `CylindN`)
- Functions/Methods: `snake_case` (e.g., `get_tp`, `reseed`)
- Variables: `snake_case` (e.g., `maxq`, `minq`)
- Constants: `UPPER_CASE` (e.g., `PI`, `HALF_PI`, `NEG_COSINE`)
- Private members: No underscore prefix (project convention)

### Docstrings (reST with Google-style)
```python
def get_tp_recursive(n: int) -> np.ndarray:
    """Recursively calculates the table-lookup of the mapping function for n.

    Args:
        n (int): The dimension.

    Returns:
        np.ndarray: The table-lookup of the mapping function.
    """
```
- Module docstrings: Raw strings `r"""..."""`
- Use `Args:` and `Returns:` sections
- Include `Examples:` for classes with usage

### Error Handling
- Use `assert` for input validation: `assert n >= 2`
- Prefer explicit validation over try/except
- Use type annotations to document expected types

### Code Formatting
- Formatter: Black (max_line_length = 256)
- Indentation: 4 spaces
- No trailing whitespace

## Project Structure

```
sphere-n/
├── src/sphere_n/
│   ├── __init__.py       # Package init, version handling
│   ├── sphere_n.py       # Main generators (Sphere3, SphereN)
│   ├── cylind_n.py       # Cylinder mapping generators
│   └── discrep_2.py      # Dispersion measure
├── tests/
│   ├── conftest.py       # Pytest fixtures
│   ├── test_sphere3.py   # Sphere3 tests
│   └── test_sp_n.py      # SphereN and CylindN tests
├── experiments/          # Demo scripts
├── docs/                 # Sphinx docs
├── setup.cfg             # Config (pytest, flake8, etc.)
├── .pre-commit-config.yaml
└── requirements.txt
```

## Testing Patterns

- Test files: `test_*.py` in `tests/`
- Test functions: `test_*()`
- Floating-point: Use `pytest.approx()`

```python
from pytest import approx
from sphere_n.sphere_n import Sphere3

def test_sphere3() -> None:
    sgen = Sphere3([2, 3, 5])
    sgen.reseed(0)
    res = sgen.pop()
    assert res[0] == approx(0.2913440162992141)
```

**Test Config**: Coverage enabled (`--cov sphere_n --cov-report term-missing`), test path: `tests/`

## Pre-commit Hooks
- `isort` - Sort imports
- `black` - Format code
- `flake8` - Lint code
- `trailing-whitespace`, `end-of-file-fixer`, etc.

## Dependencies

**Runtime**: `importlib-metadata`, `numpy`, `lds-gen`
**Testing**: `pytest`, `pytest-cov`, `scipy`, `rich`, `numba`

## Common Patterns

```python
# Cached functions
from functools import cache

@cache
def get_tp_recursive(n: int) -> np.ndarray:
    # ...

# Abstract base classes
from abc import ABC, abstractmethod

class SphereGen(ABC):
    @abstractmethod
    def pop(self) -> List[float]:
        raise NotImplementedError

# Doctests
if __name__ == "__main__":
    import doctest
    doctest.testmod()
```

## Important Notes

1. **Type hints**: Required for all new code
2. **Docstrings**: Public functions/classes need Args/Returns
3. **Black formatting**: Run before committing (or use pre-commit)
4. **Python version**: Minimum 3.10 required
5. **Package layout**: Uses `src/` layout (setuptools find_namespace:)
6. **Test coverage**: Write tests for new functionality
