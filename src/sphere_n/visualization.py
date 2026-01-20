"""Visualization helpers for sphere-n generators.

This module provides utilities for visualizing points generated
by sphere-n generators on 2D and 3D spheres.
"""

from typing import List

import numpy as np


def plot_2d_projection(
    points: List[List[float]], projection: str = "xy", ax=None, **kwargs
) -> None:
    """Plot 2D projection of high-dimensional sphere points.

    Args:
        points: List of points (each a list of coordinates).
        projection: Which 2D projection to use ('xy', 'xz', 'yz', etc.).
        ax: Matplotlib axis (creates new if None).
        **kwargs: Additional arguments for scatter plot.

    Example:
        >>> from sphere_n import SphereN
        >>> sgen = SphereN([2, 3, 5, 7])
        >>> points = sgen.pop_batch(100)
        >>> plot_2d_projection(points, projection='xy')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )

    points_array = np.array(points)

    # Map projection to coordinate indices
    proj_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in proj_map:
        raise ValueError(f"Invalid projection: {projection}")
    idx1, idx2 = proj_map[projection]

    x = points_array[:, idx1]
    y = points_array[:, idx2]

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x, y, **kwargs)
    ax.set_xlabel(f"Coordinate {idx1}")
    ax.set_ylabel(f"Coordinate {idx2}")
    ax.set_title(f"2D Projection ({projection}) - {len(points)} points")
    ax.grid(True, alpha=0.3)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_3d_projection(points: List[List[float]], ax=None, **kwargs) -> None:
    """Plot 3D projection of sphere points.

    Args:
        points: List of 3D points (each with 4 coordinates).
        ax: Matplotlib 3D axis (creates new if None).
        **kwargs: Additional arguments for scatter plot.

    Example:
        >>> from sphere_n import Sphere3
        >>> sgen = Sphere3([2, 3, 5])
        >>> points = sgen.pop_batch(100)
        >>> plot_3d_projection(points)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for 3D visualization. "
            "Install with: pip install matplotlib"
        )

    points_array = np.array(points)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    elif not hasattr(ax, "zaxis"):
        raise ValueError("Axis must be a 3D axis")

    x = points_array[:, 0]
    y = points_array[:, 1]
    z = points_array[:, 2]

    ax.scatter(x, y, z, **kwargs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Projection - {len(points)} points")
    ax.grid(True, alpha=0.3)

    if hasattr(ax, "figure"):
        ax.figure.tight_layout()
        plt.show()


def plot_distribution_comparison(
    lds_points: List[List[float]],
    random_points: List[List[float]],
    ax=None,
    title: str = "LDS vs Random Distribution",
) -> None:
    """Compare point distribution between LDS and random sampling.

    Args:
        lds_points: Points from low-discrepancy sequence.
        random_points: Points from random sampling.
        ax: Matplotlib axis (creates new if None).
        title: Plot title.

    Example:
        >>> from sphere_n import SphereN
        >>> import numpy as np
        >>> sgen = SphereN([2, 3, 5, 7])
        >>> lds_points = np.array(sgen.pop_batch(100))
        >>> random_points = np.random.randn(100, 4)
        >>> random_points /= np.linalg.norm(random_points, axis=1)[:, None]
        >>> plot_distribution_comparison(lds_points.tolist(), random_points.tolist())
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )

    lds_array = np.array(lds_points)
    rand_array = np.array(random_points)

    # Project to 2D for visualization
    lds_x, lds_y = lds_array[:, 0], lds_array[:, 1]
    rand_x, rand_y = rand_array[:, 0], rand_array[:, 1]

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(lds_x, lds_y, alpha=0.6, label="LDS")
        ax1.set_title("Low-Discrepancy Sequence")
        ax1.grid(True, alpha=0.3)

        ax2.scatter(rand_x, rand_y, alpha=0.6, label="Random")
        ax2.set_title("Random Sampling")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.legend()
        plt.tight_layout()
        plt.show()
    else:
        ax.scatter(lds_x, lds_y, alpha=0.6, label="LDS")
        ax.scatter(rand_x, rand_y, alpha=0.6, label="Random")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)


def animate_points(
    generator, n_points: int = 50, interval: int = 100, projection: str = "3d", **kwargs
) -> None:
    """Create animated visualization of point generation.

    Args:
        generator: Sphere generator instance (Sphere3, SphereN, or CylindN).
        n_points: Number of points to animate.
        interval: Animation interval in milliseconds.
        projection: '3d' or 2D projection ('xy', 'xz', etc.).
        **kwargs: Additional arguments for scatter plot.

    Example:
        >>> from sphere_n import Sphere3
        >>> sgen = Sphere3([2, 3, 5])
        >>> sgen.reseed(42)
        >>> animate_points(sgen, n_points=100)
    """
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for animation. "
            "Install with: pip install matplotlib"
        )

    generator.reseed(42)

    if projection == "3d":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame: int) -> List[float]:
            point = generator.pop()
            ax.clear()
            x, y, z = point[0], point[1], point[2]
            ax.scatter(x, y, z, c=[frame], s=20)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_title(f"Point {frame + 1} of {n_points}")
            return point

        animation.FuncAnimation(fig, update, frames=n_points, interval=interval)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame: int) -> List[float]:
            point = generator.pop()
            ax.clear()
            x, y = point[0], point[1]
            ax.scatter(x, y, c=[frame], s=20)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(f"Point {frame + 1} of {n_points}")
            return point

        animation.FuncAnimation(fig, update, frames=n_points, interval=interval)
        plt.show()
