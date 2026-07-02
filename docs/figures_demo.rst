Figures Demo
============

Auto-generated figures demonstrating sphere-n functionality.

.. plot:: examples/plot_sphere3_points.py

.. plot:: examples/plot_sphere_projection.py

The plot inline directive
-------------------------

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   x = np.linspace(0, 2 * np.pi, 100)
   plt.plot(x, np.sin(x))
   plt.title("Simple Sine Wave")
   plt.grid(True, alpha=0.3)
