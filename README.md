# 1DLayerModeling

> 当前文档对应版本：**v1.2.0**

用于**一维法向入射层状结构**（多层材料 + 零厚度界面弹簧）的频域前向建模工具。当前实现基于**动态刚度法**（dynamic stiffness method），重点面向：

- 有限厚度多层结构的 1D 法向纵波响应
- 界面法向弹簧刚度对频谱的影响
- 左右半无限介质端接条件下的反射、透射与输入阻抗计算

当前版本可输出：

- 节点位移响应
- 反射系数 `R(ω)`
- 透射位移比 `T_u(ω)`
- 输入阻抗 `Z_in(ω)`
- 界面位移跳量 `Δu(ω)`
- 无耗、实阻抗边界下的功率反射率、透射率与功率平衡

---

## 1. 物理模型与边界定义

### 1.1 单层 `Layer`

每一层采用 1D 纵向波模型。

- 波速：`c = sqrt(E / ρ)`
- 波阻抗：`Z = ρ c`
- 波数：`k = ω / c`

层两端位移自由度为 `u_L, u_R`。当前端口力约定下，单层 2×2 动态刚度矩阵为：

`K_layer = ω Z [[cot(kh), -csc(kh)], [-csc(kh), cot(kh)]]`

当 `sin(kh)` 在共振奇点附近过小时，代码会加入极小复扰动以保持数值可解。这个处理是**数值正则化**，不是物理损耗模型。

### 1.2 界面 `InterfaceSpring`

当前 `main` 分支采用**显式有限刚度界面**。若界面法向刚度为 `K_int`，其 2×2 刚度矩阵为：

`K_spring = K_int [[1, -1], [-1, 1]]`

注意：

- 多层结构必须显式传入 `len(layers) - 1` 个有限界面弹簧。
- 当前版本不再支持用 `None / inf` 表示 perfect interface。
- 若要逼近刚接，可使用足够大的有限刚度，并通过测试检查收敛。

### 1.3 结构本体矩阵与端接矩阵

`LaminatedStack.assemble_structure_matrix(ω)` 返回的是**结构本体**的全局动态刚度矩阵 `K_struct(ω)`，尚未接入左右半空间。

之后再施加左右端接：

- `K[0,0] += iωZ_left`
- `K[-1,-1] += iωZ_right`
- `rhs[0] += 2 iω Z_left a_inc`

因此当前求解得到的 `R / T_u / Z_in` 是：

**“有限层状体 + 左右半无限介质端接” 的加载后响应**，不是裸系统本征谱。

### 1.4 端口参考面

左端输出定义在：

**左半空间与第一层左表面的交界面**

右端输出定义在：

**最后一层右表面与右半空间的交界面**

因此：

- `u[0]` 是左边界面位移
- `R = (u_left - a_inc) / a_inc` 是左端口反射系数
- `T_u = u_right / a_inc` 是右端口透射位移比

这些量都是**端口面量**，不是半空间内部远场探头信号。

---

## 2. 左右半空间介质自定义

### 2.1 `HalfSpaceMedium`

新增了 `HalfSpaceMedium` 类型，用来显式描述左右半空间端接介质。可用两种方式定义：

1. 通过 `density + wave_speed`
2. 直接给定 `acoustic_impedance`

示例：

```python
from layered1d import HalfSpaceMedium

water = HalfSpaceMedium(density=1000.0, wave_speed=1480.0, name="Water")
steel = HalfSpaceMedium(density=7850.0, wave_speed=5900.0, name="Steel")
water_eq = HalfSpaceMedium.from_impedance(1000.0 * 1480.0, name="Water-equivalent")
```

### 2.2 兼容旧接口

求解器同时支持：

```python
result = stack.solve_sweep(freqs, left_medium=water, right_medium=steel)
```

和原有的标量阻抗接口：

```python
result = stack.solve_sweep(freqs, left_medium_impedance=1.48e6, right_medium_impedance=4.63e7)
```

两种接口只能二选一。代码内部会解析出对应端口阻抗。

---

## 3. 代码结构

### 3.1 当前模块

- `layered1d/media.py`
  - `HalfSpaceMedium`：左右半空间介质定义与阻抗解析
- `layered1d/model.py`
  - `Layer`：层材料与层内场恢复
  - `InterfaceSpring`：零厚度法向弹簧界面
  - `Connectivity`：DOF 拓扑
  - `LaminatedStack`：结构装配、边界施加、单频/扫频求解
- `layered1d/solver.py`
  - `FrequencyResponseResult`：扫频结果容器与后处理
- `examples/basic_demo.py`
  - 三层结构 + 可自定义左右半空间示例
- `tests/test_physics_consistency.py`
  - 物理一致性与覆盖率增强测试

### 3.2 当前职责分层

`LaminatedStack` 目前已拆分为：

- `_build_connectivity()`：建立 DOF/节点拓扑
- `assemble_structure_matrix()`：仅装配结构本体矩阵
- `_apply_boundary_conditions()`：施加左右半空间端接与左入射激励
- `_recover_scattering_outputs()`：恢复 `R / T_u / Z_in`
- `_compute_interface_jumps()`：提取界面位移跳量
- `solve_frequency_point()`：单频求解
- `solve_sweep()`：扫频求解

这种拆法已经具备继续演化为“对象层 + 端接层”的基础。

---

## 4. 快速开始

### 4.1 安装

```bash
pip install -e .
```

### 4.2 运行示例

```bash
python examples/basic_demo.py
```

示例会在 `examples/outputs/<timestamp>/` 下生成：

- `reflection_magnitude_comparison.png`
- `input_impedance_comparison.png`

当前示例显式创建左右半空间：

```python
left_medium = HalfSpaceMedium(density=1000.0, wave_speed=1480.0, name="Water")
right_medium = HalfSpaceMedium(density=7850.0, wave_speed=5900.0, name="Steel")
```

---

## 5. 结果对象与后处理

`solve_sweep()` 返回 `FrequencyResponseResult`，包含：

- `reflection_coefficient`
- `transmission_displacement_ratio`
- `input_impedance`
- `interface_jumps`
- `left_boundary_impedance`
- `right_boundary_impedance`

并提供便捷属性：

- `reflection_magnitude`
- `reflection_phase`
- `input_impedance_magnitude`
- `interface_jump_magnitude`
- `power_reflectance`
- `power_transmittance`
- `power_balance`

其中：

`power_balance = |R|^2 + (Z_right / Z_left) |T_u|^2`

对**无耗、实阻抗边界**情形，应满足 `power_balance ≈ 1`。

---

## 6. 物理一致性验证

当前测试覆盖了以下关键检查：

### 6.1 模型核心物理检查

- 动刚矩阵低频静态极限
- 阻抗匹配单层零反射
- 无耗能量守恒
- 反向结构的传输功率互易性
- 大刚度界面向刚接参考收敛

### 6.2 接口与实现检查

- `HalfSpaceMedium` 与标量阻抗接口等价
- `q_from_amplitudes()` 与 `amplitudes_from_boundary_displacements()` 往返一致
- `field()` 恢复出的边界值和速度定义一致
- 构造器参数校验正确
- 共振奇点附近正则化后矩阵仍为有限值

运行测试：

```bash
python -m unittest discover -s tests -v
```

---

## 7. 当前模型边界与后续建议

当前模型是：

- 1D 法向入射
- 仅纵向波
- 有限厚度分层体
- 零厚度法向弹簧界面
- 左右半空间阻抗端接

因此它**不是**：

- 导波色散模型
- 任意角入射模型
- 含剪切、模态转换、各向异性板理论的完整模型
- 含换能器、耦合层、探头电学链条的完整测量模型

后续更合理的演化路线是：

1. 将“结构本体算子”与“端接/观测模型”进一步解耦。
2. 增加刚性端、自由端、频率相关阻抗等边界模型。
3. 增加损耗与频散。
4. 增加功率流、吸收、条件数、奇点附近稳健性监测。
5. 在验证充分后，再上升到可辨识性分析与后验推断。

---

## 8. 许可证

当前仓库尚未声明许可证。若计划开源分发，建议补充 `LICENSE` 文件。
