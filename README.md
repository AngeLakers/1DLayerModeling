# 1DLayerModeling

用于**层状半无限 / 横向无限介质中的法向平面纵波**频域前向建模工具。当前实现基于动态刚度法，面向：

- 有限厚度层状结构在左右半无限介质端接下的法向平面纵波响应
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
from layered1d import PowerLawAttenuation

aluminum = Material(
    density=2700.0,
    young_modulus=70e9,
    poisson_ratio=0.33,
    name="Aluminum",
)

polymer = Material(
    density=1200.0,
    young_modulus=3.0e9,
    poisson_ratio=0.40,
    attenuation=PowerLawAttenuation(
        alpha_ref=0.10,
        ref_frequency_hz=20e6,
        power=1.0,
        unit="dB/mm",
    ),
    name="Polymer",
)
```

注意：

- 当前 `Material` 默认按**各向同性固体**解释
- `longitudinal_wave_speed` 不是 `sqrt(E/ρ)`
- 当前实现使用的是层法向平面纵波速度

```text
M = E(1-ν) / ((1+ν)(1-2ν))
longitudinal_wave_speed = sqrt(M / ρ)
```

- 因此这里的 `young_modulus` **不能**再被理解成平面应变 / 横向受限有效纵向模量 `M` 或 `c11`
- `poisson_ratio` 现在是计算层内纵波速度的必要参数，不应再默认偷设为 `0`
- 当前法向平面波求解器真正参与层内计算的是 `density + young_modulus + poisson_ratio`，以及可选的 `attenuation`
- `attenuation` 表示层内衰减规律；当前实现包含 `ConstantAttenuation(alpha_np_per_m)` 和 `PowerLawAttenuation(alpha_ref, ref_frequency_hz=20e6, power=1.0, unit="Np/m")`
- `ConstantAttenuation(alpha_np_per_m)` 是常数幅值衰减，单位为 `Np/m`
- `PowerLawAttenuation(alpha_ref, ref_frequency_hz, power, unit)` 是频率相关幅值衰减
- `attenuation_alpha` 仍作为兼容旧写法的快捷参数，等价于 `ConstantAttenuation(attenuation_alpha)`
- `attenuation_law` 仍作为旧别名保留，但推荐新代码使用 `attenuation`
- 若 `attenuation`、`attenuation_law` 和 `attenuation_alpha` 都为 `None`，则层内传播按无耗处理
- 幂律衰减按 `alpha(f)=alpha_ref_Np_per_m*(f/ref_frequency_hz)**power` 计算，所有模型最终统一输出 `Np/m`
- 当 `unit="dB/mm"` 时，幅值衰减换算为 `alpha_Np/m = alpha_dB/mm * ln(10) / 20 * 1000`
- 在当前求解器的右行传播约定 `exp(-j k z)` 下，衰减通过 `k = k_real - j alpha` 引入
- 衰减是层内传播损耗，不是界面阻尼，也不是当前反演目标
- `notes` 目前仍主要用于组织化管理和后续扩展

另外，`Material` 还提供：

- `shear_modulus`
- `longitudinal_modulus`
- `shear_wave_speed`
- `longitudinal_wave_speed`
- `impedance`

`Layer` 会代理这些常用材料属性，因此 `layer.longitudinal_modulus`、
`layer.shear_modulus`、`layer.shear_wave_speed` 与 `layer.material` 上的同名属性一致。

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

旧写法仍然可用，但现在必须显式给出 `poisson_ratio`：

```python
layer = Layer(
    thickness=1.0e-3,
    density=2700.0,
    young_modulus=70e9,
    poisson_ratio=0.33,
    name="Al-1",
)
```

但当前更推荐：

```python
layer = Layer.from_material(thickness=1.0e-3, material=aluminum, name="Al-1")
```

或：

```python
layer = Layer(thickness=1.0e-3, material=aluminum, name="Al-1")
```

如果你同时传 `material` 和 `density / young_modulus / poisson_ratio`，代码会直接报错。

兼容接口保留策略：

- `Layer(...)` 直接传 `density / young_modulus / poisson_ratio` 的旧构造方式暂时保留，但会给出 `FutureWarning`；衰减模型稳定后再决定是否移除
- `Layer.wave_speed` 和 `HalfSpaceMedium.wave_speed` 作为旧别名保留，但会给出 `FutureWarning`；新代码应使用 `longitudinal_wave_speed`
- `Layer.from_material(...)`、`HalfSpaceMedium.from_impedance(...)` 和 `Material` 派生属性继续保留
- `notes` 继续作为材料元数据保留，不参与当前数值求解
- `FrequencyResponseResult.raw_solutions` 继续保留为诊断 / 回归检查数据；常规分析优先使用 `reflection_coefficient`、`input_impedance`、`interface_jumps`、`power_balance` 等结构化结果

---

## 3. 左右半空间介质

`HalfSpaceMedium` 仍然用于左右边界：

```python
from layered1d import HalfSpaceMedium

left_medium = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1480.0, name="Water")
right_medium = HalfSpaceMedium(density=7850.0, longitudinal_wave_speed=5900.0, name="Steel")
```

这里的 `longitudinal_wave_speed` 是你**直接指定**给边界半空间的纵波速度。
它不是从 `E, ν` 自动反推的材料对象。

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

- `layered1d/attenuation.py`
  - `ConstantAttenuation(alpha_np_per_m)`：常数幅值衰减规律，单位 `Np/m`
  - `PowerLawAttenuation(alpha_ref, ref_frequency_hz, power, unit)`：频率幂律幅值衰减规律，支持 `Np/m` 和 `dB/mm`
  - `AttenuationLaw`：衰减规律接口
- `layered1d/materials.py`
  - `Material`：各向同性固体层材料对象，并持有可选衰减规律
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
  - 无损耗基础示例：多层结构、零厚度界面弹簧、反射谱与输入阻抗
- `examples/constant_attenuation_demo.py`
  - 常数衰减机制示例：对比 0、20、80 `Np/m`
- `examples/power_law_attenuation_demo.py`
  - 频率幂律衰减机制示例：输出 `alpha(f)` 曲线、反射、输入阻抗、界面位移跳量与功率平衡
- `examples/attenuation_demo.py`
  - 旧入口提示脚本；推荐直接运行拆分后的两个衰减 demo
- `tests/test_physics_consistency.py`
  - 物理一致性与接口兼容性测试

示例可从仓库根目录用模块方式运行：

```bash
python -m examples.basic_demo
python -m examples.constant_attenuation_demo
python -m examples.power_law_attenuation_demo
```

---

## 5. 测试

运行：

```bash
python -m unittest discover -s tests -v
```

当前测试应覆盖：

- `Material` 派生纵波速度、横波速度与阻抗
- `Layer.from_material(...)` 与 legacy 构造方式等价
- `attenuation_alpha=None` 与 `attenuation_alpha=0.0` 的无耗等价性
- `ConstantAttenuation` 被 `Material` 持有后能驱动 `Layer.wavenumber(...)`
- `PowerLawAttenuation` 的参考频率、频率趋势、`dB/mm` 到 `Np/m` 幅值换算和非法参数校验
- `PowerLawAttenuation.alpha(omega)` 与 `np_per_m(frequency_hz)` 的角频率 / 频率入口一致性
- 有耗层复波数、传播因子衰减和功率平衡下降
- 低频静态极限对应平面应变 / 横向受限条件下的有效纵向刚度 `M / h`
- 阻抗匹配零反射
- 介质对象 / 标量阻抗等价
- 无耗功率守恒
- 反向传输功率互易
- 大刚度界面收敛
- 奇点正则化
- 振幅往返一致性
- 场恢复边界一致性
- 构造器参数校验
- `HalfSpaceMedium` 的阻抗直设、`from_impedance(...)` 工厂方法和 `wave_speed` legacy alias 警告
- 半空间介质非法输入、速度别名冲突和阻抗不一致校验
- `Layer` 厚度、material / legacy 参数互斥、`wave_speed` legacy alias 警告和边界阻抗参数校验
- 零入射振幅下输入阻抗返回无穷大的边界情况

---

## 6. 当前边界

当前模型是：

- 法向入射平面纵波
- 有限厚度分层体
- 横向无限或等效横向受限层内状态
- 零厚度法向弹簧界面
- 左右半空间阻抗端接
- 经验型层内传播衰减，可为常数或频率幂律，通过复波数 `k = omega/c - j alpha(f)` 引入

因此它不是：

- 导波色散模型
- 任意角入射模型
- 含剪切、模态转换、各向异性板理论的完整模型
- 含换能器、耦合层、电学链条的完整测量模型
- 复模量、频散、界面阻尼或 Biot 多孔介质耗散模型

---

## 7. 后续建议

如果后面继续扩展，更合理的路是：

1. 保持 `Material` 作为层内材料的一等对象
2. 把损耗、频散首先挂到 `Material`
3. 将“结构本体算子”和“端接/观测模型”进一步拆开
4. 再上升到可辨识性分析和后验推断

## 8. 数值稳健性测试

现有测试已经覆盖材料参数派生、低频极限、阻抗匹配、无耗功率守恒、大刚度界面收敛和关键参数校验。后续面向反演或极端参数场景时，应继续补充数值稳健性测试。

建议优先覆盖：

- 非极点频率处的有限差分一致性
- 极点邻域的 no-NaN、warning 或可控失败行为
- 极端界面刚度下的求解稳定性
- 线性方程残差
- 大刚度极限向刚性连接参考收敛
- 小刚度极限向脱粘或自由界面参考收敛

矩阵条件数适合作为诊断信息，但不宜作为唯一硬性判据。更可靠的判断应同时结合残差、有限性、功率平衡和物理极限收敛。
