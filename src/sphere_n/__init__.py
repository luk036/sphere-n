r"""sphere-n package for generating low-discrepancy sequences on n-dimensional spheres.

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

from importlib.metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name: str = "sphere-n"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
