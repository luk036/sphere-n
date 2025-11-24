"""sphere-n package for generating low-discrepancy sequences on n-dimensional spheres.

.. svgbob::
   :align: center

   n-Sphere Visualization:
   
   +------------------+
   |    S^n           |
   |   +----------+   |
   |  /          /|   |
   | +----------+ |   |
   | |    O     | |   |
   | |   / \    | |   |
   | |  *   *   | |   |
   | | *     *  | |   |
   | |  *   *   | |   |
   | |   * *    | |   |
   | +----------+ |   |
   |  \          \|   |
   |   +----------+   |
   +------------------+

"""

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.9`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name: str = "sphere-n"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
