# 1DLayerModeling

用于**一维法向入射层状结构**的频域前向建模工具。当前实现基于动态刚度法，面向：

- 有限厚度多层结构的 1D 法向纵波响应
- 零厚度界面弹簧对频谱的影响
- 左右半无限介质端接条件下的反射、透射与输入阻抗计算

当前版本支持两类对象：

- `layered1d.materials.Material`：层内材料参数对象
- `HalfSpaceMedium`：左右半空间端接介质对象

这两者不是一回事。
`Material` 管的是**层本体**。
`HalfSpaceMedium` 管的是**边界端接**。

---

## 1. 当前推荐的建模方式

### 1.1 先定义材料

```python
from layered1d.materials import Material

aluminum = Material(
    density=2700.0,
    young_modulus=70e9,
    name="Aluminum",
    poisson_ratio=0.33,
)

polymer = Material(
    density=1200.0,
    young_modulus=3.0e9,
    name="Polymer",
    poisson_ratio=0.40,
    notes="Only density and young_modulus are used in the current 1D solver.",
)
```

注意：

- 当前 1D 求解器真正参与计算的只有 `density` 和 `young_modulus`
- `poisson_ratio`、`attenuation_alpha`、`notes` 目前主要用于组织化管理和后续扩展
- 这里的 `young_modulus` 在当前模型语义下可理解为**1D 有效纵向模量**

### 1.2 再用材料构造层

```python
from layered1d import Layer

layers = [
    Layer.from_material(thickness=1.0e-3, material=aluminum, name="Al-1"),
    Layer.from_material(thickness=0.2e-3, material=polymer, name="Polymer"),
    Layer.from_material(thickness=1.0e-3, material=aluminum, name="Al-2"),
]
```

这比直接把 `density=...`、`young_modulus=...` 在每一层里重复写一遍更清楚。

---

## 2. 向后兼容

旧写法仍然可用：

```python
layer = Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9, name="Al-1")
```

但当前更推荐：

```python
layer = Layer.from_material(thickness=1.0e-3, material=aluminum, name="Al-1")
```

或：

```python
layer = Layer(thickness=1.0e-3, material=aluminum, name="Al-1")
```

如果你同时传 `material` 和 `density / young_modulus`，代码会直接报错。

---

## 3. 左右半空间介质

`HalfSpaceMedium` 仍然用于左右边界：

```python
from layered1d import HalfSpaceMedium

left_medium = HalfSpaceMedium(density=1000.0, wave_speed=1480.0, name="Water")
right_medium = HalfSpaceMedium(density=7850.0, wave_speed=5900.0, name="Steel")
```

然后：

```python
result = stack.solve_sweep(
    freqs,
    left_medium=left_medium,
    right_medium=right_medium,
)
```

---

## 4. 代码结构

- `layered1d/materials.py`
  - `Material`：层材料参数对象
- `layered1d/media.py`
  - `HalfSpaceMedium`：半无限边界介质
- `layered1d/model.py`
  - `Layer`
  - `InterfaceSpring`
  - `Connectivity`
  - `LaminatedStack`
- `layered1d/solver.py`
  - `FrequencyResponseResult`
- `examples/basic_demo.py`
  - 使用 `Material + Layer.from_material(...)` 的示例
- `tests/test_physics_consistency.py`
  - 物理一致性与接口兼容性测试

---

## 5. 测试

运行：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- `Material` 派生波速与阻抗
- `Layer.from_material(...)` 与旧构造方式等价
- 低频静态极限
- 阻抗匹配零反射
- 介质对象 / 标量阻抗等价
- 无耗功率守恒
- 反向传输功率互易
- 大刚度界面收敛
- 奇点正则化
- 振幅往返一致性
- 场恢复边界一致性
- 构造器参数校验

---

## 6. 当前边界

当前模型是：

- 1D 法向入射
- 仅纵向波
- 有限厚度分层体
- 零厚度法向弹簧界面
- 左右半空间阻抗端接

因此它不是：

- 导波色散模型
- 任意角入射模型
- 含剪切、模态转换、各向异性板理论的完整模型
- 含换能器、耦合层、电学链条的完整测量模型

---

## 7. 后续建议

如果后面继续扩展，更合理的路是：

1. 保持 `Material` 作为层内材料的一等对象
2. 将损耗、频散先挂到 `Material` 上
3. 将“结构本体算子”和“端接/观测模型”进一步拆开
4. 再上升到可辨识性分析和后验推断
