# Changelog

## Unreleased

基于 `v1.2.2-1`，补齐测试运行依赖和边界校验覆盖。

### Added

- 新增 `tests/test_validation_edges.py`，覆盖输入校验和防御性分支：
  - `AdhesiveLayerPrior` 非法标量、非法区间、越界泊松比和负适用性比例
  - `ConstantAttenuation.alpha(omega)` 负角频率
  - `Material` 非法材料参数、非法衰减入口和衰减规律返回负值 / NaN
  - `HalfSpaceMedium` 必填和可选密度 / 纵波速度校验
  - `Layer` legacy 构造器不可转换泊松比
  - `LaminatedStack` 接口数量校验和 `FrequencyResponseResult.layer_field(...)` 形状稳定性
- `requirements.txt` 明确 `pytest>=7.0` 是运行完整测试套件的必需依赖。

### Changed

- `requirements.txt` 将构建依赖和测试依赖拆分说明，避免把 `pytest` 记录为可选开发体验。

### Removed

- 移除 `requirements.txt` 中 `pytest` 可选依赖的说明。

### Validation

- 当前验证通过：

```bash
conda activate multilayer_model
python -m pytest
```

## v1.2.2-1

基于 `v1.2.2`，新增 benchmark、胶层先验和零厚度界面适用性检查。

### Added

- 新增 `layered1d/adhesives.py`：
  - `AdhesiveLayerPrior` 可复用胶层 / 聚合物层文献先验
  - `A1_DEFAULT_ADHESIVE_PRIOR` 和 `NOA60_HALDREN_2019_PRIOR`
  - A1 默认胶层 `Material` / `Layer` 工厂函数
- 新增 `layered1d/model_checks.py`：
  - `zero_thickness_interface_ratio(...)`
  - `classify_zero_thickness_interface_ratio(...)`
  - `check_zero_thickness_interface_applicability(...)`
  - `check_layer_as_zero_thickness_interface(...)`
- 新增 `scripts/test.ps1`，统一执行项目测试。
- 新增 benchmark 和模型检查测试：
  - `tests/test_adhesive_priors.py`
  - `tests/test_benchmark_regressions.py`
  - `tests/test_model_checks.py`

### Changed

- README 补充胶层先验、零厚度 `K_N` 适用性阈值、reduced-model 判定和标准测试流程。
- `layered1d.__init__` 导出胶层先验和模型检查 API。
- `CHANGELOG.md` 的验证命令从直接调用 `unittest` 改为 `.\scripts\test.ps1`。

### Validation

- 新增和扩展测试覆盖：
  - transfer-matrix 和 acoustic-impedance benchmark 对照
  - 界面刚度刚接 / 自由界面极限
  - phase unwrap 和 group-delay 回归
  - 既有 `Layer` 的零厚度界面适用性判定
  - A1 默认胶层和 NOA60 文献先验

## v1.2.2

基于 `v1.2.1`，主要加入层内衰减模型，并重新整理 demo 结构。

### Added

- 新增 `layered1d/attenuation.py`：
  - `AttenuationLaw` 衰减规律接口
  - `ConstantAttenuation(alpha_np_per_m)` 常数幅值衰减，单位 `Np/m`
  - `PowerLawAttenuation(alpha_ref, ref_frequency_hz, power, unit)` 频率幂律幅值衰减，支持 `Np/m` 和 `dB/mm`
- `ConstantAttenuation` 和 `PowerLawAttenuation` 均提供兼容角频率入口 `alpha(omega)`，内部换算为 `np_per_m(omega / (2*pi))`。
- `Material` 新增 `attenuation` 参数，用于持有层内传播衰减规律。
- `attenuation_alpha` 继续作为旧写法兼容入口，等价于 `ConstantAttenuation(attenuation_alpha)`。
- `attenuation_law` 作为旧别名保留，并给出 `FutureWarning`。
- `Layer.wavenumber()` 现在会通过 `Material.attenuation_np_per_m(frequency_hz)` 引入层内传播衰减，当前传播约定下使用 `k = k_real - j alpha`。
- 新增 `examples/constant_attenuation_demo.py`，只展示常数衰减机制，包含 `0/20/80 Np/m` 三组。
- 新增 `examples/power_law_attenuation_demo.py`，只展示频率幂律衰减机制，包含 `alpha(f)` 曲线图和响应图。

### Changed

- `examples/basic_demo.py` 恢复为无损耗基础 baseline，不再默认给 polymer 引入非零常数衰减。
- `examples/attenuation_demo.py` 不再混合常数衰减和幂律衰减出图，改为提示运行拆分后的两个 demo。
- README 补充衰减模型接口、`dB/mm` 到 `Np/m` 幅值换算、复波数符号约定和 demo 结构说明。
- `__all__` 导出衰减模型相关类型，便于从 `layered1d` 顶层直接导入。

### Validation

- 新增和扩展测试覆盖：
  - `ConstantAttenuation` 被 `Material` 持有后驱动 `Layer.wavenumber(...)`
  - `PowerLawAttenuation` 的参考频率、频率趋势、`dB/mm` 到 `Np/m` 幅值换算
  - `PowerLawAttenuation.alpha(omega)` 与 `np_per_m(frequency_hz)` 的一致性
  - 非法衰减参数、负频率和负角频率校验
  - 有耗层复波数、传播因子衰减和功率平衡下降
- 当前验证通过：

```bash
.\scripts\test.ps1
```

## v1.2.1

### Changed

- 修正纵波定义：层内 `longitudinal_wave_speed` 明确为横向无限 / 横向受限层状介质中的法向平面纵波速度。
- README 与代码注释去除细杆一维杆波语义，改用有效纵向模量 `M` 表述。
- 低频静态极限说明改为平面应变 / 横向受限条件下的有效纵向刚度 `M / h`。

### Fixed

- `Layer` legacy 构造器现在会显式检查 `poisson_ratio` 缺失或非有限值，并返回清晰错误信息。
- `Layer` 补齐 `longitudinal_modulus`、`shear_modulus`、`shear_wave_speed` 代理属性，与 `Material` API 保持一致。

### Validation

- 测试通过：

```bash
python -m unittest discover -s tests -v
```

## v1.2.0

基于 `main (v1.1.1)`、`feature/custom-halfspace-physics-tests` 和 `copilot/analyze-test-coverage` 的清理整合版本。

### Added

- 新增 `HalfSpaceMedium`，支持显式定义左右半空间介质。
- 新增 `__version__ = "1.2.0"`。
- 新增 `.gitignore`，忽略 `__pycache__`、`.coverage`、`.pytest_cache/`。
- 吸收 coverage 分支中的有效测试逻辑，并统一到 `unittest` 体系。

### Changed

- `solve_frequency_point()` / `solve_sweep()` 同时支持：
  - 标量阻抗输入
  - `HalfSpaceMedium` 对象输入
- `FrequencyResponseResult` 增加：
  - `left_boundary_impedance`
  - `right_boundary_impedance`
  - `power_reflectance`
  - `power_transmittance`
  - `power_balance`
- `examples/basic_demo.py` 改为显式定义左右半空间，不再把水介质写死。
- `README.md` 已与当前 `main` 行为和 v1.2.0 接口对齐，并删除不再适用的 perfect-interface 旧描述。

### Validation

- 合并后测试通过：

```bash
python -m unittest discover -s tests -v
```

- 当前整合版测试覆盖：
  - 低频静态极限
  - 阻抗匹配零反射
  - 介质对象 / 标量阻抗接口等价
  - 无耗功率守恒
  - 反向传输功率互易
  - 大刚度界面收敛
  - 构造器校验
  - 振幅往返一致性
  - 场恢复边界一致性
  - 奇点正则化可解性
