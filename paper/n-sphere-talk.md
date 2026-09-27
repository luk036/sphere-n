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
- **Multivariate empirical mode decomposition** [@rehman2010multivariate] -- direction vectors on higher-dimensional spheres via the **cylindrical mapping**
- **Filter bank design** [@mandic2011filter] -- the same cylindrical mapping underlies its filter-bank analysis
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

# Previous Work

## Related Work

- **Low-discrepancy sequences & quasi-Monte Carlo**
  - van der Corput (1935) [@vandercorput1935]; Halton (1960) [@halton1960]; Sobol' (1967) [@sobol1967]
- **Equidistribution & designs on the sphere**
  - Cui & Freeden [@cui1997equidistribution]; Brauchart et al. [@brauchart2012qmc]; Saff & Kuijlaars [@saff1997]
- **Cylindrical & Hopf mappings**
  - Wong, Luk & Heng [@wong1997sampling]; Mitchell [@mitchell2008sampling]; Yershova et al. [@yershova2010generating]; Shoemake [@shoemake1992]

## Related Work (cont.)

- **Random sampling on $S^n$**
  - normalise a Gaussian vector; rejection method [@marsaglia1972; @fishman1996]
- **Spherical codes & applications**
  - Grassmannian beamforming [@strohmer2003grassmannian; @love2003grassmannian]
  - space-time codes from spherical codes [@utkovski2006construction]
  - multivariate EMD via cylindrical-mapped LDS [@rehman2010multivariate; @mandic2011filter]

## The Gap Addressed Here

- No previous construction is **uniform + deterministic + incremental** on $S^n$ for **arbitrary $n$**
- Cylindrical and Hopf mappings exploit structure special to $S^2$ / $S^3$
- Spherical designs and energy minimisation are **not incremental** (adding a point changes the configuration)
- Random sampling is **not deterministic**
- This work: van der Corput + **recursive tabulated inverse CDF**

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

## Why Cylindrical Mapping Works Only for $S^2$

- Write a point of $S^n$ as $p = (\cos\theta_n,\ \sin\theta_n\, u)$, $u \in S^{n-1}$
- The surface element separates into a height part and a spherical part:
  $$d^nA = \sin^{n-1}\theta_n\, d\theta_n\, dA_{n-1}(u)$$
- Substituting the height $z = \cos\theta_n$:
  $$d^nA = (1-z^2)^{(n-2)/2}\, dA_{n-1}(u)\, dz$$
- The height weight $(1-z^2)^{(n-2)/2}$ is constant **only when $n = 2$**
- $S^2$: $d^2A = d\varphi\, dz$ -- uniform $z$ and $\varphi$ is exact (the cylindrical mapping)
- $n \ge 3$: uniform height is **biased** (over-samples the poles)
  - draw $\theta_n$ by inverting the CDF of $\sin^{n-1}\theta_n$, i.e. $f_{n-1}$

## Recursive Construction on $S^n$

- Hyperspherical coordinates:
  - $x_0 = \cos\theta_n$
  - $x_1 = \sin\theta_n \cos\theta_{n-1}$
  - $\dots$
  - $x_n = \sin\theta_n \sin\theta_{n-1} \cdots \sin\theta_1$
- Surface element:
  $$d^nA = \sin^{n-1}\theta_n \sin^{n-2}\theta_{n-1}\cdots\sin\theta_2\, d\theta_1\cdots d\theta_n$$
- It factorises, but the inverse has **no closed form** for $m \ge 2$
- Peel off one angle at a time: $p_n = [\,\cos\theta_n,\; \sin\theta_n \cdot p_{n-1}\,]$

## How to Generate the Point Set

- Let $f_m(\theta) = \int_0^\theta \sin^m\varphi \, \mathrm{d}\varphi$, defined recursively:
  $$
  f_m(\theta) = \begin{cases}
    \theta & m = 0, \\
    -\cos\theta & m = 1, \\
    \tfrac{1}{m}\bigl(-\cos\theta\,\sin^{m-1}\theta + (m-1) f_{m-2}(\theta)\bigr) & m \ge 2.
  \end{cases}
  $$
- Angle $\theta_j$ ($j = 2,\dots,n$) carries weight $\sin^{j-1}\theta_j$: map $\mathrm{vdc}(k, b_j)$ onto $f_{j-1}$
- Invert numerically: $\theta_j = f_{j-1}^{-1}(t_j)$ by **table lookup**; $f_0, f_1$ are closed forms
- Assemble $p_j = [\cos\theta_j,\ \sin\theta_j \cdot p_{j-1}]$ up to $p_n$

## Table Lookup: Numerical Mechanics

- **Resolution**: every table is on a **300-point** grid of $[0,\pi]$
  - spacing $h = \pi/299 \approx 1.05\times10^{-2}$ rad
- **Interpolation**: binary search + linear interpolation
  - $O(\log 300) \approx 9$ comparisons, clamped at the endpoints
- **Precision**: piecewise-linear error $O(h^2)$
  - max angular error $\approx 3.4 \,/\, 2.7 \,/\, 2.3 \times 10^{-4}$ rad for $m = 2, 3, 4$
- **Memory**: 300 doubles $\approx 2.4$ kB per table
  - $O(n)$ tables, under $20$ kB for $n \le 5$; built lazily and cached

## Implementation

- Van der Corput generator -- uniform values in $[0,1]$
- Interpolation routines -- invert the cached tables
- `SphereGen` -- abstract interface (`pop`, `reseed`)
- `Sphere3` -- explicit generator for $S^3$; `SphereN` -- recursive chain bottoming out at `Sphere3`
- Any dimension, limited only by memory

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
- **Random**: normalise a Gaussian vector (uniform on $S^n$)
- **LDS**: `SphereN` vs the `CylindN` baseline
- Left: ours, right: random
:::
::: {.column width="55%"}
![](res_compare.pdf){width="100%"}
:::
::::

## Dispersion at $N = 600$ (30 random trials)

| Sphere (bases) | Random | CylindN | SphereN |
|---|---|---|---|
| $S^3$ (2,3,5) | $0.845 \pm 0.041$ | 0.659551 | **0.650145** |
| $S^4$ (2,3,5,7) | $1.090 \pm 0.038$ | 1.050584 | **0.912591** |
| $S^5$ (2,3,5,7,11) | $1.252 \pm 0.041$ | 1.358791 | **1.035655** |

- Lower is better; `SphereN` is lowest on every sphere
- Beats the random **mean** by more than one standard deviation

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
