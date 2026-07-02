"""
2D Projection of Sphere Points
===============================

Compares XY and XZ projections of 3-sphere points.
"""
import matplotlib.pyplot as plt
import numpy as np
from sphere_n.sphere_n import Sphere3

sgen = Sphere3([2, 3, 5])
sgen.reseed(42)
points = np.array(sgen.pop_batch(300))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.6, c='steelblue')
plt.xlabel('Coordinate 0')
plt.ylabel('Coordinate 1')
plt.title('XY Projection')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(points[:, 0], points[:, 2], s=10, alpha=0.6, c='lightcoral')
plt.xlabel('Coordinate 0')
plt.ylabel('Coordinate 2')
plt.title('XZ Projection')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
