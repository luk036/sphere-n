# Motivation

## Low-Discrepancy Sequences

- A **deterministic** sequence of points that fills a space more uniformly than random sampling
- Three desirable properties:
  - **Uniformity** -- points are spread evenly, minimising clusters and gaps
  - **Determinism** -- the sequence is reproducible from a fixed starting point
  - **Incrementality** *(the key property)* -- points can be appended while the whole set stays evenly distributed
- Drawback: generation becomes slower as the number of points grows

## Applications of LDS on $S^n$

- **Robot motion planning** on $S^3$ and $SO(3)$ [@yershova2010generating] -- for path planning and attitude control
- **Wireless coding** -- spherical codes and MIMO codebooks [@utkovski2006construction]
- **Multivariate empirical mode decomposition** [@rehman2010multivariate] -- more accurate signal models
- **Filter bank design** [@mandic2011filter] -- more precise filter parameters
- **Statistical and machine learning** -- deterministic coverage of a normalised parameter space

## The Challenge in Higher Dimensions

- Sampling the 3-D sphere ($n = 2$) is well understood
- Little is known for $n > 2$ -- the *curse of dimensionality*
- The **cylindrical mapping does not generalise** to higher dimensions
- Goal: one construction on $S^n$ that is uniform, deterministic and incremental

# Overview of Low-Discrepancy Sequences

## The van der Corput Sequence

:::: {.columns}
::: {.column width="45%"}
- The one-dimensional low-discrepancy sequence on $[0,1]$
- Constructed by reversing the base-$b$ digits of the integers, with $b$ usually prime
- Introduced by the Dutch mathematician **Johannes van der Corput** (1935)
:::
::: {.column width="55%"}
![](vdcorput.pdf){width="100%"}
:::
::::

## Unit Circle $S^1$

:::: {.columns}
::: {.column width="50%"}
- Treat the vdC value as an angle:
  $$\theta = 2\pi \cdot \mathrm{vdc}(k, b)$$
- $$[x, y] = [\cos\theta, \sin\theta]$$
- Incremental and uniform by construction
:::
::: {.column width="50%"}
![](circle.pdf){width="80%"}
:::
::::

## Halton Sequence on $[0,1]^n$

:::: {.columns}
::: {.column width="55%"}
- Combine $n$ van der Corput sequences with **coprime** bases:
  $$[x_1, \dots, x_n] = [\mathrm{vdc}(k, b_1), \dots, \mathrm{vdc}(k, b_n)]$$
- Uniform over the unit hypercube
- Basis of quasi-Monte Carlo (QMC) methods
:::
::: {.column width="45%"}
![](halton3d.pdf){width="95%"}
:::
::::

## Unit Sphere $S^2$ -- Cylindrical Mapping

:::: {.columns}
::: {.column width="55%"}
- Two vdC values give $z$ and $\varphi$:
  - $z = 2\,\mathrm{vdc}(k, b_2) - 1 \in [-1, 1]$
  - $\varphi = 2\pi\,\mathrm{vdc}(k, b_1) \in [0, 2\pi)$
- Radius of the horizontal circle: $r = \sqrt{1 - z^2}$
- $$[x, y, z] = [\,r\cos\varphi,\; r\sin\varphi,\; z\,]$$
- Used in computer graphics [@wong1997sampling]
:::
::: {.column width="45%"}
![](sphere.pdf){width="85%"}
:::
::::

## $S^3$ and $SO(3)$ -- Hopf Fibration

- Hopf coordinates [@yershova2010generating]:
  - $x_1 = \cos(\theta/2)\cos(\psi/2)$
  - $x_2 = \cos(\theta/2)\sin(\psi/2)$
  - $x_3 = \sin(\theta/2)\cos(\varphi + \psi/2)$
  - $x_4 = \sin(\theta/2)\sin(\varphi + \psi/2)$
- $S^3$ is a principal circle bundle over $S^2$
- Works for $S^3$ only: the cylindrical mapping does **not** extend to $S^n$ for $n > 2$

# Our Approach

## Uniform Sampling on a Unit Disk

:::: {.columns}
::: {.column width="55%"}
- Surface element: $dA = r \, dr \, d\theta$
- Integrating gives $\int r\,dr = \tfrac{1}{2} r^2$, so the radial weight is $r$
- Cancel it with the **inverse**: use $r = \sqrt{\mathrm{vdc}(k, b_2)}$
- $\theta = 2\pi\,\mathrm{vdc}(k, b_1)$
- $[x, y] = [r\cos\theta, r\sin\theta]$
:::
::: {.column width="45%"}
![](disk.pdf){width="85%"}
:::
::::

## Recursive Construction on $S^n$

- Polar coordinates:
  - $x_0 = \cos\theta_n$
  - $x_1 = \sin\theta_n \cos\theta_{n-1}$
  - $\dots$
  - $x_n = \sin\theta_n \sin\theta_{n-1} \cdots \sin\theta_1$
- The surface element factorises, but its **inverse has no closed form** for $n \ge 2$
- Key idea:
  $$p_n = [\,\cos\theta_n,\; \sin\theta_n \cdot p_{n-1}\,]$$
  building $S^n$ from a point on $S^{n-1}$

## How to Generate the Point Set

- Let $f_j(\theta) = \int \sin^j\theta \, \mathrm{d}\theta$, defined recursively:
  $$
  f_j(\theta) = \begin{cases}
    \theta & j = 0, \\
    -\cos\theta & j = 1, \\
    \tfrac{1}{n}\bigl(-\cos\theta\,\sin^{j-1}\theta + (n-1) f_{j-2}(\theta)\bigr) & j \ge 2.
  \end{cases}
  $$
- $f_j$ is monotone on $(0,\pi)$; map $\mathrm{vdc}(k, b_j)$ onto $[f_j(0), f_j(\pi)]$
- Invert numerically: $\theta_j = f_j^{-1}(t_j)$ by **table lookup**
- Then recurse with $p_n = [\cos\theta_n, \sin\theta_n \cdot p_{n-1}]$

## Implementation

- `SphereGen` -- abstract interface (`pop`, `reseed`)
- `Sphere3` -- explicit generator for $S^3$
- `SphereN` -- recursive chain that bottoms out at `Sphere3`
- Inverse tables are built once by linear interpolation and **cached**

## Reference Implementations

- Four sibling projects, all implementing the same generators:
  - `lds-gen` (Python) -- locks, `pop_batch()`
  - `lds-rs` (Rust) -- `AtomicU64`, `SphereN`
  - `lds-cpp` (C++20) -- `constexpr`, precomputed tables
  - `lds-gen-cpp` (C++20) -- includes the `sphere_n` module
- Core generators in all four: `VdCorput`, `Halton`, `Circle`, `Disk`, `Sphere`, `Sphere3Hopf`, `HaltonN`
- The recursive `Sphere3`/`SphereN` exist in Python, Rust and `lds-gen-cpp` only

## Cross-Language Verification and Performance

- With a fixed seed the core generators are **bit-identical** across languages
- The recursive sphere generators agree to 15+ decimal places; one coordinate differed at the 16th (machine epsilon)
- Nanoseconds per point:
  - `Sphere3` ($S^3$): C++ 657, Rust 1035, Python 7749
  - `SphereN` ($S^4$): Rust 1607, C++ 1665, Python 12153
  - `SphereN` ($S^5$): Rust 2020, C++ 2312, Python 48443
- C++ templates win for shallow generators; Rust's lock-free counter wins for deep recursion

# Numerical Experiments

## Dispersion Measure

- Generate $N$ points and build their convex hull (`scipy.spatial.ConvexHull`)
- Dispersion is the spread of neighbour distances:
  $$\max_{a \in \mathcal{N}(b)} \{D(a,b)\} - \min_{a \in \mathcal{N}(b)} \{D(a,b)\}, \qquad D(a,b) = \sqrt{1 - a^\mathsf{T} b}$$
- Lower dispersion means a more uniform point set

## Random vs LDS ($N = 600$)

:::: {.columns}
::: {.column width="45%"}
- **Random**: normalise a Gaussian vector, which is uniform on $S^n$
- **LDS**: the recursive generator
- Left: ours, right: random
:::
::: {.column width="55%"}
![](res_compare.pdf){width="100%"}
:::
::::

## Results: $S^3$ vs Hopf Coordinate Method

![](res_hopf.pdf){width="72%"}

## Results: $S^3$ vs Cylindrical Mapping

![](res-S3-cylin.pdf){width="72%"}

## Results: $S^4$ vs Cylindrical Mapping

![](res-S4-cylin.pdf){width="72%"}

## Results: $S^5$ vs Cylindrical Mapping

![](res-S5-cylin.pdf){width="72%"}

# Conclusions

## Conclusions

- A uniform, deterministic and **incremental** LDS on $S^n$
- Built from the van der Corput sequence plus a recursive spherical construction
- Outperforms random sampling, especially when the number of points is small
- Superior to the Hopf method on $S^3$ and to cylindrical mapping on $S^4$ and $S^5$

## Future Work

- Monte Carlo integration, optimisation and machine learning
- Faster table lookup or closed-form approximations for large $n$
- Extensions to $SO(n)$ and other manifolds

## References {#references .allowframebreaks}
