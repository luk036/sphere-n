# 低差异序列在超球面上的生成及其应用

## 摘要

本文探讨了在 n 维超球面上生成低差异序列的方法。低差异序列在数值积分、优化和仿真等领域具有重要价值。本文详细介绍了超球面采样方法的期望特性，包括均匀性、确定性和增量性。随后介绍了基于 van der Corput 序列的 n 维超球面低差异序列生成方法。我们提供了该算法及其在 sphere_n 库中的实现的详细说明。此外，本文展示了为评估所提出方法性能而进行的数值实验结果，包括与随机生成序列和其他方法（如 Hopf 坐标方法和圆柱坐标方法）的比较。

## 1. 引言

低差异序列在数学、计算机科学和工程的众多领域中发挥着关键作用。这些序列提供比随机采样更均匀的点分布，使其成为数值积分、优化和仿真应用的宝贵工具。虽然低差异采样方法在较低维度（特别是在三维空间球面上）已得到广泛研究，但对高维度高效采样技术的需求日益增长。

### 1.1 研究动机

低差异序列相对于传统随机采样方法具有几个优势：

1. **均匀性**：低差异序列本质上旨在实现点的**均匀分布**。这意味着点在整个采样空间中均匀分布，最小化高浓度或稀疏区域。
2. **确定性**：与真正的随机采样方法不同，低差异序列是**确定性的**。这意味着对于给定的起始点和参数，生成的序列始终相同，使结果具有可重复性。
3. **增量性**：这是描述的**最重要优势**和定义特征。增量性意味着序列可以**逐步增长**。
   * **动态分布**：每次向序列中添加新点时，**整体分布保持相对均匀**。这对于事先不知道所需样本数量的应用至关重要。
   * **应用灵活性**：在机器人学等领域，人们可能无法精确知道需要多少样本，可能在几个点后停止或切换策略。低差异序列通过在每个新点处**保持均匀分布**来适应这一点。

随着点数的增加，生成它们的过程变得缓慢是主要缺点。

在转向更高维度时，必须解决许多挑战才能促进进展。维数诅咒是挑战之一。随着维数的增加，实现均匀分布变得越来越具有挑战性。虽然三维球面的采样方法已得到充分研究，但高维度的理论仍处于发展阶段。

本文提出了一种基于 van der Corput 序列的在 $S^n$ 上生成低差异序列的方法。该方法的目的是解决与高维采样相关的固有挑战，同时保持与低差异序列相关的期望特性。

### 1.2 研究目标

本项目的主要目标包括：

- 研究低差异序列的数学基础，特别是其在超球面上的生成方法
- 实现并优化 n 维超球面上的低差异序列生成算法
- 通过数值实验验证所提出方法的有效性
- 探索低差异序列在超球面采样中的实际应用

## 2. 文献综述

### 2.1 低差异序列基础

#### 2.1.1 van der Corput 序列

van der Corput 序列是一种数学序列，用于生成 0 到 1 之间均匀分布的一系列数字。它是低差异序列的基础，用于在区间 [0,1] 上生成均匀分布的点。该序列通过特定的基（通常是质数）构建。

算法描述：
```
对于整数 k，将 k 转换为以 base 为底的数字表示
将数字顺序颠倒，转换为小数部分
例如：k=5, base=2
- 5 的二进制表示是 101
- 颠倒顺序得到 101
- 转换为小数 0.101 = 0.625
```

#### 2.1.2 Halton 序列

Halton 序列是一种确定性的点序列，在多维空间中均匀分布。该序列由 Halton 和 Rutishauser 在 20 世纪 60 年代开发。Halton 序列是通过组合两个或多个以不同质数为基的 van der Corput 序列构建的。

#### 2.1.3 Sphere 序列

Sphere 序列将低差异序列扩展到球面。通过将 van der Corput 序列与球面坐标相结合，可以生成在球面上均匀分布的点。

### 2.2 高维球面采样方法

#### 2.2.1 Hopf 纤维化方法

对于 S³ 和 SO(3)，研究人员提出使用 Hopf 纤维化生成均匀增量网格。该方法利用了四维球面的复结构和几何性质。

#### 2.2.2 圆柱映射方法

圆柱映射方法通过将高维球面投影到低维空间来生成点序列。这种方法简单但可能在高维时产生不均匀性。

#### 2.2.3 递归极坐标方法

递归极坐标方法通过递归地构建高维球面来生成均匀分布的点。这是 sphere_n 库采用的主要方法。

## 3. 方法论

### 3.1 Sphere-n 库的算法实现

sphere-n 库提供了一套用于在 n 维球面上生成低差异序列的工具。其核心算法基于递归结构和极坐标变换。

#### 3.1.1 递归极坐标变换

在 n 维球面上，点可以用极坐标表示：

- $x_0 = \cos\theta_n$
- $x_1 = \sin\theta_n \cos\theta_{n-1}$
- $x_2 = \sin\theta_n \sin\theta_{n-1} \cos\theta_{n-2}$
- $x_3 = \sin\theta_n \sin\theta_{n-1} \sin\theta_{n-2} \cos\theta_{n-3}$
- $\cdots$
- $x_{n-1} = \sin\theta_n \sin\theta_{n-1} \sin\theta_{n-2} \cdots \cos\theta_1$
- $x_n = \sin\theta_n \sin\theta_{n-1} \sin\theta_{n-2} \cdots \sin\theta_1$

#### 3.1.2 球面元素

n 维球面的微分元素为：

$$d^nA = \sin^{n-2}(\theta_{n-1})\sin^{n-1}(\theta_{n-2})\cdots \sin(\theta_{2})\,d\theta_1 \, d\theta_2\cdots d\theta_{n-1}$$

#### 3.1.3 点集生成算法

1. $p_0 = [\cos\theta_1, \sin\theta_1]$ 其中 $\theta_1 = 2\pi\cdot\mathrm{vdc}(k,b_1)$
2. 设 $f_j(\theta) = \int\sin^j\theta \mathrm{d}\theta$，其中 $\theta\in (0,\pi)$
3. $f_j(\theta)$ 可以递归定义为：
   $$
   f_j(\theta) =
   \begin{cases}
     \theta          & \text{如果 } j = 0 , \\
     -\cos\theta     & \text{如果 } j = 1 , \\
     (1/j)( -\cos\theta \sin^{j-1}\theta + (j-1)\int\sin^{j-2}\theta \mathrm{d}\theta) & \text{其他情况}.
   \end{cases}
   $$
4. 将 $\mathrm{vdc}(k,b_j)$ 均匀映射到 $f_j(\theta)$：$t_j = f_j(0) + (f_j(\pi) - f_j(0)) \mathrm{vdc}(k,b_j)$
5. 设 $\theta_j = f_j^{-1}(t_j)$
6. 递归定义 $p_n$：$p_n = [\cos\theta_n, \sin\theta_n \cdot p_{n-1}]$

### 3.2 SphereN 类实现

SphereN 类是 n 维球面生成器的核心实现：

```python
class SphereN(SphereGen):
    def __init__(self, base: List[int]) -> None:
        n = len(base) - 1
        assert n >= 2
        self.vdc = VdCorput(base[0])
        if n == 2:
            self.s_gen = Sphere(base[1:3])
        else:
            self.s_gen = SphereN(base[1:])
        self.n = n
        tp = get_tp(n)
        self.range = tp[-1] - tp[0]

    def pop(self) -> List[float]:
        if self.n == 2:
            ti = HALF_PI * self.vdc.pop()  # 映射到 [t0, tm-1]
            xi = np.interp(ti, F2, X)
            cosxi = math.cos(xi)
            sinxi = math.sin(xi)
            return [sinxi * s for s in self.s_gen.pop()] + [cosxi]

        vd = self.vdc.pop()
        tp = get_tp(self.n)
        ti = tp[0] + self.range * vd  # 映射到 [t0, tm-1]
        xi = np.interp(ti, tp, X)
        sinphi = math.sin(xi)
        return [xi * sinphi for xi in self.s_gen.pop()] + [math.cos(xi)]
```

### 3.3 性能优化策略

sphere-n 库采用多种性能优化策略：

1. **缓存机制**：使用 `@cache` 装饰器缓存计算值，避免重复计算
2. **查找表**：预先计算查找表（X、NEG_COSINE、SINE、F2 等）以加速插值操作
3. **递归结构**：通过较低维度的球面来构建较高维度的球面，简化实现

## 4. 实验与结果

### 4.1 实验设计

为评估 sphere-n 库中实现的低差异序列生成方法的性能，我们进行了数值实验。实验比较了多种生成方法：

1. 随机采样方法
2. SphereN 生成器（本文方法）
3. CylindN 生成器（圆柱映射方法）
4. Sphere3Hopf 生成器（Hopf 坐标方法）

### 4.2 评估指标

我们使用分散度（dispersion）作为评估均匀性的主要指标：

- 分散度大致通过每两个相邻点之间的最大距离和最小距离的差来测量：
  $$
  \max_{a \in \mathcal{N}(b)} \{D(a,b)\} - \min_{a \in \mathcal{N}(b)} \{ D(a, b) \}
  $$
  其中 $D(a,b) = \sqrt{1 - a^\mathsf{T} b}$

- 使用 scipy.spatial.ConvexHull 函数构建凸包

### 4.3 实验结果

实验结果表明，在不同维度的超球面上，sphere-n 库实现的方法在均匀性方面优于随机采样方法和其他比较方法。

在 S³（球面维度为 4）上，我们的方法比 Hopf 坐标方法表现出更好的点分布均匀性。在 S⁴（球面维度为 5）上，我们的方法比圆柱映射方法表现出更好的点分布均匀性。

### 4.4 性能分析

随着点数的增加，sphere-n 库的方法保持了良好的均匀性，特别是在点数较少时，我们的方法在均匀性和确定性方面表现出明显优势。

## 5. 应用

### 5.1 机器人运动规划

在高维空间如 $S^3$ 和 $SO(3)$ 中，Halton 序列提供均匀分布的点集，适用于机器人路径规划和姿态控制。这允许优化计算效率和运动轨迹的准确性。

### 5.2 无线通信编码

在 MIMO 系统中，球面编码采用 Halton 序列生成的点作为码字，以增强信号传输的稳定性和抗干扰能力，并改善数据传输率和质量。

### 5.3 多元经验模态分解

在多元经验模态分解中，Halton 序列可用于构建更精确的信号模型，提高信号处理的准确性。

### 5.4 滤波器组设计

在滤波器组设计的上下文中，利用 Halton 序列有助于构建更精确的滤波器参数，从而增强信号处理的准确性。

### 5.5 计算机图形学

在计算机图形学中，均匀分布的点可用于光线追踪、全局光照计算和纹理映射等应用。

### 5.6 数值积分

在数值积分中，超球面上的均匀点分布可以提高高维积分计算的精度和效率。

## 6. 结论与未来工作

### 6.1 结论

本文提供了对 n 维球面低差异采样方法的全面讨论。我们介绍了一种基于 van der Corput 序列的在 n 维球面上生成低差异序列的建议方法，解决了与高维采样相关的挑战，同时保持了所需的特性。

sphere-n 库的实现证明了该方法的效率和生成具有高度均匀性点集的能力。数值实验表明，所提出的方法在均匀性方面优于随机序列和其他现有方法，特别是在点数较少的情况下。

### 6.2 未来工作

建议的方法代表了低差异序列领域的一个有价值的贡献，在许多应用中具有潜力。该方法既高效又能生成具有高度均匀性的点集。该方法可用于生成用于各种应用的点集，包括蒙特卡洛模拟、优化和机器学习。此外，该方法易于实现，可以被研究人员和从业人员使用。

未来工作的可能方向包括：

1. GPU 加速：利用 GPU 并行计算能力优化点生成算法
2. 算法扩展：实现和分析其他低差异序列算法，如 Sobol 序列和 Faure 序列
3. 应用集成：将 sphere-n 库集成到实际应用中，如机器人仿真和数值积分系统
4. 理论分析：进一步研究生成序列的数学性质，如不规则性度量和收敛性

## 参考文献

1. Yershova, A., Jain, S., Lavalle, S. M., & Mitchell, J. C. (2010). Generating uniform incremental grids on SO(3) using the Hopf fibration. *The International Journal of Robotics Research*, 29(7), 801-812.

2. Wong, K. Y., & Niblack, W. (2003). An efficient and effective algorithm for the computation of the spatial distribution of the electromagnetic field in a microstrip structure. *IEEE Transactions on Microwave Theory and Techniques*, 51(8), 1820-1825.

3. Mandic, D. P., & Goh, S. L. (2009). Complex-valued prediction and filtering. *IEEE Transactions on Signal Processing*, 57(12), 4560-4572.

4. Rehman, N., & Mandic, D. P. (2010). Multivariate empirical mode decomposition. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 466(2117), 1291-1302.

5. Mitchell, J. S. (2008). Optimization algorithms and applications. *CRC Press*.

6. Utkovski, Z., & Mathar, R. (2006). Construction of codebooks for spherical codes. *IEEE Transactions on Information Theory*, 52(10), 4714-4720.

7. Fishman, G. S. (1996). Monte Carlo: Concepts, Algorithms, and Applications. *Springer-Verlag*.

8. Sobol, I. M. (1967). Distribution of points in a cube and approximate evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics*, 7(4), 86-112.

9. Niederreiter, H. (1992). Random Number Generation and Quasi-Monte Carlo Methods. *SIAM*.

10. Kuipers, L., & Niederreiter, H. (2012). Uniform Distribution of Sequences. *Courier Corporation*.
