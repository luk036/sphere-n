"""
3-Sphere Point Distribution
===========================

Generates and visualizes points on a 3-sphere using low-discrepancy sequences.
"""
import matplotlib.pyplot as plt
import numpy as np
from sphere_n.sphere_n import Sphere3

sgen = Sphere3([2, 3, 5])
sgen.reseed(42)
points = np.array(sgen.pop_batch(200))

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           c=points[:, 3], cmap='viridis', s=15, alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('200 Points on S\u00b3 (colored by 4th coordinate)')
fig.tight_layout()
