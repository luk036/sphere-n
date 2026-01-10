# IFLOW.md - sphere-n 项目上下文

## 项目概述

`sphere-n` 是一个用于在 n 维球面上生成低差异序列（Low-discrepancy sequences）的 Python 库。该库的主要目的是在不同维度的球面上生成均匀分布的点，这在计算机图形学、数值积分和蒙特卡洛模拟等领域具有重要价值。

该项目基于 PyScaffold 4.5 构建，使用 MIT 许可证。它依赖于 `lds-gen` 库来生成低差异序列的底层组件。

## 核心组件

### 主要模块

1. **sphere_n.py** - 实现了 n 维球面的生成器
   - `SphereGen` - 所有球面生成器的抽象基类
   - `Sphere3` - 3 维球面序列生成器
   - `SphereN` - n 维球面序列生成器
   - `get_tp` 函数 - 计算映射函数的查找表

2. **cylind_n.py** - 使用圆柱映射生成 n 维球面上点的模块
   - `CylindGen` - 圆柱映射生成器的抽象基类
   - `CylindN` - 使用圆柱映射的低差异序列生成器

3. **discrep_2.py** - 包含用于计算分散度测量的函数

## 架构和逻辑流

### 核心算法

1. **VdCorput 序列生成器** - 生成在 0 到 1 之间均匀分布的数字序列
2. **映射函数** - 将生成的数字映射到球面表面
3. **递归结构** - 通过较低维度的球面来构建较高维度的球面

### 生成流程

- 从构建 `SphereN` 对象开始，该对象使用 `Sphere3`（对于 3D）或递归过程来生成较低维度的球面以构建更高维度的球面
- 在生成点时，使用 VdCorput 序列获得基础数字，然后通过涉及正弦、余弦和插值的一系列变换将其映射到球面上

### 性能优化

- 使用 `@cache` 装饰器来优化性能，通过存储和重用计算值
- 预先计算的查找表（X、NEG_COSINE、SINE、F2 等）用于加速插值操作

## 文件结构

```
src/sphere_n/
├── __init__.py
├── sphere_n.py         # n 维球面生成器
├── cylind_n.py         # 圆柱映射生成器
├── discrep_2.py        # 分散度测量函数
└── py.typed            # 标记包为类型提示兼容
```

## 测试和验证

- `tests/test_sp_n.py` 包含了用于评估不同球面点生成方法的完整测试套件
- 包含针对随机生成和低差异序列生成的比较测试
- 验证生成点的维度和归一化

## 依赖关系

### 运行时依赖
- `lds-gen` - 低差异序列生成器库
- `numpy` - 数值计算
- `decorator` - 装饰器支持

### 测试依赖
- `pytest` - 测试框架
- `scipy` - 用于凸包计算
- `numpy` - 数值计算

## 使用示例

### SphereN 生成器
```python
from sphere_n.sphere_n import SphereN

sgen = SphereN([2, 3, 5, 7])
sgen.reseed(0)
for _ in range(1):
    print(sgen.pop())
```

### CylindN 生成器
```python
from sphere_n.cylind_n import CylindN

cgen = CylindN([2, 3, 5, 7])
cgen.reseed(0)
for _ in range(1):
    print(cgen.pop())
```

## 开发约定

- 项目使用 Python 3.10+
- 遵循 PEP 8 代码风格（通过 flake8 检查）
- 使用类型提示
- 包含 doctest 示例
- 使用 pytest 进行测试
- 使用 setuptools 进行构建

## 构建和测试命令

```bash
# 安装依赖
pip install -e .

# 运行测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov sphere_n --cov-report term-missing

# 运行 doctest
python -m doctest src/sphere_n/sphere_n.py
```
