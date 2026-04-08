# 1DLayerModeling

用于**一维法向入射层状结构**（多层材料 + 零厚度界面弹簧）的频域建模工具。当前实现基于动态刚度法（dynamic stiffness method），可计算：

- 节点位移响应
- 反射系数与透射位移比
- 输入阻抗
- 界面位移跳量（imperfect interface 开裂/滑移表征）

---

## 1. 物理模型与刚度矩阵逻辑

### 1.1 单层（Layer）动态刚度

对每一层（厚度 `h`，密度 `ρ`，模量 `E`），采用 1D 纵向波模型：

- 波速：`c = sqrt(E / ρ)`
- 波阻抗：`Z = ρ c`
- 波数：`k = ω / c`

层两端位移自由度为 `u_L, u_R`，在当前端口力约定下，单层 2×2 动态刚度矩阵为：

`K_layer = ω Z [[cot(kh), -csc(kh)], [-csc(kh), cot(kh)]]`

代码中在 `sin(kh)` 过小（共振奇点附近）时加入微小复扰动，避免数值崩溃，但不改变基本物理关系。  

### 1.2 界面弹簧（InterfaceSpring）

若界面法向刚度为 `K_int`（N/m³ 的等效离散形式，具体量纲依建模标定），其 2×2 刚度矩阵：

`K_spring = K_int [[1, -1], [-1, 1]]`

- `stiffness=None` 或 `inf`：视为完美粘接（无额外 DOF、无显式弹簧矩阵）
- 有限刚度：在界面两侧拆分为 `L/R` 两个节点，通过弹簧连接

### 1.3 全局装配

`LaminatedStack` 中全局矩阵装配分两步：

1. 累加每一层 `K_layer`
2. 对非完美界面累加 `K_spring`

通过 `_scatter_add_2x2` 统一执行局部 2×2 到全局矩阵的散装配（scatter-add），后续扩展更清晰。

### 1.4 边界条件（半无限介质辐射边界）

在结构左右端引入外介质阻抗 `Z_left, Z_right`：

- `K[0,0] += iωZ_left`
- `K[-1,-1] += iωZ_right`

左侧给定入射位移振幅 `a_inc` 时，右端无入射，仅左端激励项：

- `rhs[0] += 2 iω Z_left a_inc`

求解线性方程后：

- 反射：`R = (u_left - a_inc) / a_inc`
- 透射位移比：`T_u = u_right / a_inc`
- 输入阻抗：由左端力/速度比定义

---

## 2. 代码结构（已为后续扩展整理）

### 2.1 关键模块

- `layered1d/model.py`
  - `Layer`：层材料与场恢复
  - `InterfaceSpring`：界面弹簧
  - `Connectivity`：DOF 拓扑/节点信息
  - `LaminatedStack`：装配与求解主流程
- `layered1d/solver.py`
  - `FrequencyResponseResult`：扫频结果容器与后处理
- `examples/basic_demo.py`
  - 三层结构 + 多组界面刚度样例

### 2.2 重构后的职责分离（不改变物理逻辑）

`LaminatedStack` 内部已按“可插拔”流程拆分：

- `_build_connectivity()`：建立 DOF/节点拓扑
- `assemble_structure_matrix()`：结构本体矩阵
- `_apply_boundary_conditions()`：外介质辐射边界 + 激励
- `_recover_scattering_outputs()`：反射/透射/输入阻抗恢复
- `_compute_interface_jumps()`：界面位移跳量
- `solve_frequency_point()`：单频调度
- `solve_sweep()`：扫频调度

这种结构便于后续加入：

- 新界面模型（粘弹、频散、非线性线性化）
- 新边界模型（刚性端、自由端、阻抗频变）
- 额外输出（能流、吸收系数、层内应力峰值）

---

## 3. 快速开始

### 3.1 安装

```bash
pip install -e .
```

### 3.2 运行示例

```bash
python examples/basic_demo.py
```

示例会在 `examples/outputs/<timestamp>/` 下生成反射与输入阻抗对比图。

---

## 4. 二次开发建议

1. **新增物理元件**：优先实现“局部 2×2 元件矩阵 + DOF 映射”，再接入装配。  
2. **保持端口符号约定一致**：位移、力方向与边界激励项必须统一。  
3. **频率相关参数**：若 `E(ω)` 或 `K_int(ω)` 频变，可在对应类中增加 `dynamic_stiffness(omega)` 的频散版本。  
4. **数值稳定性**：在奇点附近建议保留微扰/正则化策略。  

---

## 5. 后续 README 可继续完善方向

- 增加“理论推导附录”（从波动方程到 2×2 动刚矩阵）
- 增加“单位检查表”（参数量纲与常见误用）
- 增加“与实验数据对比工作流”
- 增加“API 文档 + 最小可复现案例（MRE）”

---

## 6. 许可证

当前仓库尚未声明许可证。若计划开源分发，建议补充 `LICENSE` 文件。


## 7. 物理一致性验证（可复现实验）

仓库内新增了 `tests/test_physics_consistency.py`，覆盖了三个关键极限/基准场景：

1. **低频静态极限**：验证 `Layer.dynamic_stiffness(ω)` 在 `ω→0` 下收敛到 `E/h * [[1,-1],[-1,1]]`。  
2. **阻抗匹配单层反射为零**：当层与左右外介质阻抗一致时，`R≈0`。  
3. **超大界面刚度逼近完美界面**：`K_int→∞` 时结果逼近 perfect interface。  

运行方式：

```bash
PYTHONPATH=. python -m unittest tests/test_physics_consistency.py -v
```

这三个验证分别对应了：

- 动刚矩阵公式本身（解析极限）；
- 边界条件与波散射定义（物理边界正确性）；
- 界面模型连续性（从 imperfect 到 perfect 的一致性）。
