# Changelog

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
