from __future__ import annotations

import math
import unittest

import numpy as np

from layered1d import (
    ConstantAttenuation,
    HalfSpaceMedium,
    InterfaceSpring,
    LaminatedStack,
    Layer,
    PowerLawAttenuation,
)
from layered1d.materials import Material


def isotropic_E_from_normal_longitudinal_speed(
    rho: float,
    longitudinal_wave_speed: float,
    nu: float,
) -> float:
    """由法向纵波速度反算杨氏模量。"""
    return (
        rho
        * longitudinal_wave_speed
        * longitudinal_wave_speed
        * (1.0 + nu)
        * (1.0 - 2.0 * nu)
        / (1.0 - nu)
    )


class PhysicsConsistencyTests(unittest.TestCase):
    def test_material_derives_wave_speeds_and_impedance(self) -> None:
        """材料接口应正确派生模量、波速和阻抗。"""
        material = Material(
            density=2700.0,
            young_modulus=70e9,
            poisson_ratio=0.33,
            name="Aluminum",
        )
        self.assertGreater(material.longitudinal_wave_speed, material.shear_wave_speed)
        self.assertGreater(material.impedance, 0.0)
        self.assertGreater(material.longitudinal_modulus, material.young_modulus)

    def test_material_requires_poisson_ratio(self) -> None:
        """Material 应要求 poisson_ratio 必须提供且为有限数。"""
        with self.assertRaisesRegex(TypeError, "poisson_ratio"):
            Material(density=2700.0, young_modulus=70e9, name="Aluminum")
        with self.assertRaisesRegex(ValueError, "poisson_ratio must be provided"):
            Material(density=2700.0, young_modulus=70e9, poisson_ratio=None, name="Aluminum")

    def test_layer_legacy_constructor_requires_poisson_ratio(self) -> None:
        """旧版 Layer 构造应拒绝缺失或非有限 poisson_ratio。"""
        with self.assertRaisesRegex(ValueError, "poisson_ratio must be provided"):
            Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9)
        with self.assertRaisesRegex(ValueError, "poisson_ratio must be finite"):
            Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9, poisson_ratio=float("nan"))

    def test_layer_can_be_built_from_material_or_legacy_properties(self) -> None:
        """Material 路径和旧参数路径的 Layer 动态刚度应等价。"""
        material = Material(density=2700.0, young_modulus=70e9, poisson_ratio=0.33, name="Aluminum")
        layer_from_material = Layer.from_material(thickness=1.0e-3, material=material, name="Al-1")
        with self.assertWarns(FutureWarning):
            layer_legacy = Layer(
                thickness=1.0e-3,
                density=2700.0,
                young_modulus=70e9,
                poisson_ratio=0.33,
                name="Al-1",
            )
        omega = 2.0 * math.pi * 0.8e6
        np.testing.assert_allclose(layer_from_material.dynamic_stiffness(omega), layer_legacy.dynamic_stiffness(omega))
        self.assertEqual(layer_from_material.material.name, "Aluminum")
        self.assertEqual(layer_from_material.name, "Al-1")

    def test_layer_legacy_constructor_carries_material_metadata(self) -> None:
        """旧版 Layer 接口应保留材料元数据并代理材料属性。"""
        with self.assertWarns(FutureWarning):
            layer = Layer(
                thickness=1.0e-3,
                density=2700.0,
                young_modulus=70e9,
                poisson_ratio=0.33,
                attenuation_alpha=12.0,
                notes="legacy path",
                name="Al-1",
            )
        self.assertEqual(layer.poisson_ratio, 0.33)
        self.assertEqual(layer.attenuation_alpha, 12.0)
        self.assertEqual(layer.notes, "legacy path")
        self.assertGreater(layer.longitudinal_wave_speed, 0.0)
        self.assertGreater(layer.shear_wave_speed, 0.0)
        self.assertEqual(layer.longitudinal_modulus, layer.material.longitudinal_modulus)
        self.assertEqual(layer.shear_modulus, layer.material.shear_modulus)
        self.assertEqual(layer.shear_wave_speed, layer.material.shear_wave_speed)
        with self.assertWarns(FutureWarning):
            self.assertEqual(layer.wave_speed, layer.longitudinal_wave_speed)

    def test_zero_attenuation_matches_lossless_layer_response(self) -> None:
        """None 和 0.0 衰减都表示无耗，响应应一致。"""
        material_kwargs = dict(density=1200.0, young_modulus=2.2e9, poisson_ratio=0.35, name="Polymer")
        lossless_none = Material(**material_kwargs, attenuation_alpha=None)
        lossless_zero = Material(**material_kwargs, attenuation_alpha=0.0)
        layer_none = Layer.from_material(thickness=1.0e-3, material=lossless_none)
        layer_zero = Layer.from_material(thickness=1.0e-3, material=lossless_zero)
        omega = 2.0 * math.pi * 0.7e6
        self.assertEqual(layer_none.wavenumber(omega), layer_zero.wavenumber(omega))

        freqs = np.array([0.3e6, 0.9e6, 1.5e6])
        left = HalfSpaceMedium(acoustic_impedance=lossless_none.impedance)
        right = HalfSpaceMedium(acoustic_impedance=lossless_none.impedance)
        result_none = LaminatedStack([layer_none]).solve_sweep(freqs, left_medium=left, right_medium=right)
        result_zero = LaminatedStack([layer_zero]).solve_sweep(freqs, left_medium=left, right_medium=right)
        np.testing.assert_allclose(result_none.reflection_coefficient, result_zero.reflection_coefficient)
        np.testing.assert_allclose(result_none.transmission_displacement_ratio, result_zero.transmission_displacement_ratio)

    def test_material_holds_constant_attenuation_law(self) -> None:
        """Material 应持有常数衰减规律，Layer 只调用它。"""
        material = Material(
            density=1200.0,
            young_modulus=2.2e9,
            poisson_ratio=0.35,
            attenuation=ConstantAttenuation(12.0),
        )
        layer = Layer.from_material(thickness=1.0e-3, material=material)
        omega = 2.0 * math.pi * 0.7e6
        self.assertEqual(material.attenuation_alpha, 12.0)
        self.assertEqual(material.attenuation_np_per_m(0.7e6), 12.0)
        self.assertEqual(material.attenuation_coefficient(omega), 12.0)
        self.assertEqual(layer.wavenumber(omega).imag, -12.0)

    def test_material_accepts_legacy_attenuation_law_alias(self) -> None:
        """旧 attenuation_law 入口仍兼容但应提示弃用。"""
        with self.assertWarns(FutureWarning):
            material = Material(
                density=1200.0,
                young_modulus=2.2e9,
                poisson_ratio=0.35,
                attenuation_law=ConstantAttenuation(12.0),
            )
        self.assertIs(material.attenuation, material.attenuation_law)
        self.assertEqual(material.attenuation_np_per_m(1.0e6), 12.0)

    def test_material_rejects_duplicate_attenuation_configuration(self) -> None:
        """新旧衰减入口三者最多只能提供一个。"""
        with self.assertRaisesRegex(ValueError, "at most one"):
            Material(
                density=1200.0,
                young_modulus=2.2e9,
                poisson_ratio=0.35,
                attenuation_alpha=12.0,
                attenuation_law=ConstantAttenuation(12.0),
            )

    def test_power_law_attenuation_units_and_reference_frequency(self) -> None:
        """幂律衰减模型应使用 Hz 参考频率并统一输出 Np/m。"""
        law = PowerLawAttenuation(alpha_ref=20.0, ref_frequency_hz=20e6, power=1.0, unit="Np/m")
        self.assertEqual(law.np_per_m(20e6), 20.0)
        self.assertEqual(law.np_per_m(10e6), 10.0)

        db_law = PowerLawAttenuation(alpha_ref=0.10, ref_frequency_hz=20e6, power=1.0, unit="dB/mm")
        expected_ref = 0.10 * math.log(10.0) / 20.0 * 1000.0
        self.assertAlmostEqual(db_law.np_per_m(20e6), expected_ref)
        self.assertAlmostEqual(db_law.np_per_m(10e6), expected_ref * 0.5)

    def test_power_law_alpha_matches_np_per_m_at_angular_frequency(self) -> None:
        """PowerLawAttenuation.alpha(omega) 应兼容角频率入口。"""
        law = PowerLawAttenuation(alpha_ref=0.10, ref_frequency_hz=20e6, power=1.0, unit="dB/mm")
        omega = 2.0 * math.pi * 20e6
        self.assertAlmostEqual(law.alpha(omega), law.np_per_m(20e6))
        with self.assertRaisesRegex(ValueError, "omega"):
            law.alpha(-omega)

    def test_power_law_attenuation_frequency_trend_and_constant_limit(self) -> None:
        """power=0 应为常数衰减，正幂律应让高频衰减更强。"""
        constant_limit = PowerLawAttenuation(alpha_ref=30.0, ref_frequency_hz=20e6, power=0.0)
        self.assertEqual(constant_limit.np_per_m(0.0), 30.0)
        self.assertEqual(constant_limit.np_per_m(1.0e3), 30.0)
        self.assertEqual(constant_limit.np_per_m(20e6), 30.0)

        law = PowerLawAttenuation(alpha_ref=30.0, ref_frequency_hz=20e6, power=1.0)
        self.assertLess(law.np_per_m(1.0e3), law.np_per_m(20e6))
        self.assertGreater(law.np_per_m(40e6), law.np_per_m(20e6))

    def test_attenuation_law_rejects_invalid_values(self) -> None:
        """衰减规律应拒绝非法单位、频率和负参数。"""
        with self.assertRaisesRegex(ValueError, "alpha_ref"):
            PowerLawAttenuation(alpha_ref=-1.0)
        with self.assertRaisesRegex(ValueError, "ref_frequency_hz"):
            PowerLawAttenuation(alpha_ref=1.0, ref_frequency_hz=0.0)
        with self.assertRaisesRegex(ValueError, "power"):
            PowerLawAttenuation(alpha_ref=1.0, power=-1.0)
        with self.assertRaisesRegex(ValueError, "unit"):
            PowerLawAttenuation(alpha_ref=1.0, unit="dB/m")
        with self.assertRaisesRegex(ValueError, "frequency_hz"):
            PowerLawAttenuation(alpha_ref=1.0).np_per_m(-1.0)
        with self.assertRaisesRegex(ValueError, "alpha_np_per_m"):
            ConstantAttenuation(-1.0)
        with self.assertRaisesRegex(ValueError, "frequency_hz"):
            ConstantAttenuation(1.0).np_per_m(-1.0)

    def test_layer_wavenumber_uses_frequency_dependent_attenuation(self) -> None:
        """Layer 应通过 Material 按频率取得 alpha(f)。"""
        material = Material(
            density=1200.0,
            young_modulus=2.2e9,
            poisson_ratio=0.35,
            attenuation=PowerLawAttenuation(alpha_ref=20.0, ref_frequency_hz=1.0e6, power=2.0),
        )
        layer = Layer.from_material(thickness=1.0e-3, material=material)
        omega = 2.0 * math.pi * 2.0e6
        k = layer.wavenumber(omega)
        self.assertEqual(material.attenuation_np_per_m(2.0e6), 80.0)
        self.assertEqual(k.imag, -80.0)

    def test_wavenumber_rejects_negative_omega(self) -> None:
        """负角频率不是当前正频谐波响应入口。"""
        material = Material(density=1200.0, young_modulus=2.2e9, poisson_ratio=0.35)
        layer = Layer.from_material(thickness=1.0e-3, material=material)
        with self.assertRaisesRegex(ValueError, "omega"):
            layer.wavenumber(-1.0)
        with self.assertRaisesRegex(ValueError, "omega"):
            material.attenuation_coefficient(-1.0)

    def test_attenuated_propagation_factor_decays_with_depth(self) -> None:
        """右行传播因子应随传播深度衰减。"""
        material = Material(density=1200.0, young_modulus=2.2e9, poisson_ratio=0.35, attenuation_alpha=50.0)
        layer = Layer.from_material(thickness=1.0e-3, material=material)
        k = layer.wavenumber(2.0 * math.pi * 0.7e6)
        project_sign = -1.0
        self.assertLessEqual(abs(np.exp(project_sign * 1j * k * layer.thickness)), 1.0)

    def test_attenuated_layer_power_balance_is_below_lossless_case(self) -> None:
        """有耗传播应降低反射加透射功率和。"""
        lossless = Material(density=1000.0, young_modulus=1.875e9, poisson_ratio=0.25, attenuation_alpha=0.0)
        attenuated = Material(density=1000.0, young_modulus=1.875e9, poisson_ratio=0.25, attenuation_alpha=80.0)
        freqs = np.array([0.2e6, 0.8e6, 1.4e6])

        lossless_layer = Layer.from_material(thickness=1.0e-3, material=lossless)
        attenuated_layer = Layer.from_material(thickness=1.0e-3, material=attenuated)
        left = HalfSpaceMedium(acoustic_impedance=lossless.impedance)
        right = HalfSpaceMedium(acoustic_impedance=lossless.impedance)

        lossless_result = LaminatedStack([lossless_layer]).solve_sweep(freqs, left_medium=left, right_medium=right)
        attenuated_result = LaminatedStack([attenuated_layer]).solve_sweep(freqs, left_medium=left, right_medium=right)

        np.testing.assert_allclose(lossless_result.power_balance, np.ones_like(freqs), rtol=1e-9, atol=1e-9)
        self.assertTrue(np.all(attenuated_result.power_balance <= lossless_result.power_balance + 1e-12))
        self.assertTrue(np.all(attenuated_result.power_balance < 1.0))

    def test_dynamic_stiffness_matches_low_frequency_static_limit(self) -> None:
        """动态刚度低频极限应为有效纵向刚度 M/h。"""
        with self.assertWarns(FutureWarning):
            layer = Layer(
                thickness=1.0e-3,
                density=2700.0,
                young_modulus=70e9,
                poisson_ratio=0.33,
            )
        omega = 1.0
        k_dynamic = layer.dynamic_stiffness(omega)
        k_static = (layer.longitudinal_modulus / layer.thickness) * np.array(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=complex,
        )
        np.testing.assert_allclose(k_dynamic, k_static, rtol=1e-9, atol=1e-3)

    def test_dynamic_stiffness_regularizes_exact_sine_pole(self) -> None:
        """精确正弦极点经正则化后仍应有限。"""
        rho = 1.0
        nu = 0.25
        longitudinal_wave_speed = 1.0
        E = isotropic_E_from_normal_longitudinal_speed(rho, longitudinal_wave_speed, nu)
        with self.assertWarns(FutureWarning):
            layer = Layer(thickness=1.0, density=rho, young_modulus=E, poisson_ratio=nu)
        omega = math.pi * layer.longitudinal_wave_speed / layer.thickness
        k_dynamic = layer.dynamic_stiffness(omega)
        self.assertTrue(np.isfinite(k_dynamic).all())

    def test_amplitude_roundtrip_matches_boundary_displacements(self) -> None:
        """边界位移和行波幅值应可往返转换。"""
        material = Material(density=1200.0, young_modulus=2.2e9, poisson_ratio=0.35, name="Polymer")
        layer = Layer.from_material(thickness=1.5e-3, material=material)
        omega = 2.0 * math.pi * 0.7e6
        a_plus = 0.8 + 0.2j
        a_minus = -0.3 + 0.1j
        u_left, u_right = layer.q_from_amplitudes(omega, a_plus=a_plus, a_minus=a_minus)
        rec_plus, rec_minus = layer.amplitudes_from_boundary_displacements(omega, u_left=u_left, u_right=u_right)
        np.testing.assert_allclose(rec_plus, a_plus)
        np.testing.assert_allclose(rec_minus, a_minus)

    def test_field_respects_boundary_values_and_velocity_definition(self) -> None:
        """场恢复应保持边界位移和速度定义一致。"""
        with self.assertWarns(FutureWarning):
            layer = Layer(
                thickness=2.0e-3,
                density=1800.0,
                young_modulus=3.6e9,
                poisson_ratio=0.30,
            )
        omega = 2.0 * math.pi * 0.3e6
        u_left = 1.2 - 0.4j
        u_right = -0.6 + 0.7j
        z_local = np.array([0.0, layer.thickness / 2.0, layer.thickness])
        field = layer.field(omega, z_local=z_local, u_left=u_left, u_right=u_right)
        np.testing.assert_allclose(field["u"][0], u_left)
        np.testing.assert_allclose(field["u"][-1], u_right)
        np.testing.assert_allclose(field["velocity"], -1j * omega * field["u"])

    def test_interface_spring_rejects_non_finite_or_non_positive_stiffness(self) -> None:
        """界面刚度必须为有限正数。"""
        with self.assertRaisesRegex(ValueError, "must be finite"):
            InterfaceSpring(stiffness=float("nan"))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            InterfaceSpring(stiffness=0.0)

    def test_laminated_stack_constructor_validations(self) -> None:
        """层状结构必须有层，并显式给出层间界面。"""
        with self.assertRaisesRegex(ValueError, "At least one layer is required"):
            LaminatedStack(layers=[])

        with self.assertWarns(FutureWarning):
            layers = [
                Layer(thickness=0.5e-3, density=1000.0, young_modulus=2.0e9, poisson_ratio=0.30),
                Layer(thickness=0.7e-3, density=1200.0, young_modulus=2.5e9, poisson_ratio=0.35),
            ]
        with self.assertRaisesRegex(ValueError, "Explicit interfaces are required"):
            LaminatedStack(layers=layers, interfaces=[])

    def test_half_space_medium_impedance_paths_and_aliases(self) -> None:
        """半空间接口应支持阻抗直设、工厂方法和 wave_speed 旧别名。"""
        impedance_only = HalfSpaceMedium(acoustic_impedance=1.8e6, name="Z-only")
        self.assertEqual(impedance_only.impedance, 1.8e6)
        with self.assertWarns(FutureWarning):
            self.assertIsNone(impedance_only.wave_speed)

        via_factory = HalfSpaceMedium.from_impedance(impedance=2.2e6, name="factory")
        self.assertEqual(via_factory.impedance, 2.2e6)

        with self.assertWarns(FutureWarning):
            compatible_alias = HalfSpaceMedium(
                density=1000.0,
                longitudinal_wave_speed=1500.0,
                wave_speed=1500.0,
            )
        with self.assertWarns(FutureWarning):
            self.assertEqual(compatible_alias.wave_speed, compatible_alias.longitudinal_wave_speed)

    def test_half_space_medium_rejects_invalid_or_conflicting_inputs(self) -> None:
        """HalfSpaceMedium 应拒绝缺失、冲突或自相矛盾的输入。"""
        with self.assertRaisesRegex(ValueError, "Provide only one of wave_speed or longitudinal_wave_speed"):
            HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1500.0, wave_speed=1490.0)

        with self.assertRaisesRegex(ValueError, "Provide either acoustic_impedance"):
            HalfSpaceMedium(density=1000.0)

        with self.assertRaisesRegex(ValueError, "acoustic_impedance must be positive and finite"):
            HalfSpaceMedium(acoustic_impedance=0.0)

        with self.assertRaisesRegex(ValueError, "inconsistent with density"):
            HalfSpaceMedium(
                density=1000.0,
                longitudinal_wave_speed=1500.0,
                acoustic_impedance=1.0e6,
            )

    def test_layer_and_boundary_impedance_validation_edges(self) -> None:
        """Layer 和边界阻抗输入应拒绝非法值。"""
        material = Material(density=2700.0, young_modulus=70e9, poisson_ratio=0.33, name="Aluminum")
        with self.assertRaisesRegex(ValueError, "thickness must be positive and finite"):
            Layer.from_material(thickness=0.0, material=material)
        with self.assertRaisesRegex(ValueError, "Provide either material"):
            Layer(
                thickness=1.0e-3,
                density=2700.0,
                young_modulus=70e9,
                poisson_ratio=0.33,
                material=material,
            )

        stack = LaminatedStack(layers=[Layer.from_material(thickness=1.0e-3, material=material)])
        with self.assertRaisesRegex(ValueError, "Provide exactly one of left_medium_impedance or left_medium"):
            stack.solve_frequency_point(0.5e6, right_medium_impedance=1.5e6)
        with self.assertRaisesRegex(ValueError, "Provide exactly one of left_medium_impedance or left_medium"):
            stack.solve_frequency_point(
                0.5e6,
                left_medium_impedance=1.5e6,
                left_medium=HalfSpaceMedium(acoustic_impedance=1.5e6),
                right_medium_impedance=1.5e6,
            )
        with self.assertRaisesRegex(ValueError, "left boundary impedance must be positive and finite"):
            stack.solve_frequency_point(0.5e6, left_medium_impedance=-1.0, right_medium_impedance=1.5e6)

    def test_zero_incident_amplitude_returns_infinite_input_impedance(self) -> None:
        """边界速度为零时输入阻抗应为无穷大。"""
        material = Material(density=1000.0, young_modulus=1.875e9, poisson_ratio=0.25)
        stack = LaminatedStack(layers=[Layer.from_material(thickness=1.0e-3, material=material)])
        with np.errstate(divide="ignore", invalid="ignore"):
            solution = stack.solve_frequency_point(
                frequency_hz=0.5e6,
                left_medium_impedance=1.5e6,
                right_medium_impedance=1.5e6,
                incident_displacement_amplitude=0.0,
            )
        self.assertTrue(np.isinf(solution["input_impedance"]))

    def test_reflection_is_zero_for_impedance_matched_single_layer(self) -> None:
        """阻抗匹配单层结构应无反射。"""
        rho = 1000.0
        nu = 0.25
        longitudinal_wave_speed = 1500.0
        E = isotropic_E_from_normal_longitudinal_speed(rho, longitudinal_wave_speed, nu)
        material = Material(density=rho, young_modulus=E, poisson_ratio=nu, name="Matched")
        layer = Layer.from_material(thickness=1.2e-3, material=material)
        water = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1500.0, name="matched")
        stack = LaminatedStack(layers=[layer])
        freqs = np.array([0.2e6, 0.8e6, 1.4e6])
        result = stack.solve_sweep(freqs, left_medium=water, right_medium=water)
        self.assertLess(np.max(np.abs(result.reflection_coefficient)), 1e-10)

    def test_medium_object_matches_raw_impedance(self) -> None:
        """HalfSpaceMedium 对象应等价于直接传阻抗。"""
        with self.assertWarns(FutureWarning):
            layer = Layer(thickness=0.6e-3, density=2700.0, young_modulus=70e9, poisson_ratio=0.33)
        freqs = np.array([0.4e6, 1.1e6])
        water = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1480.0, name="Water")
        steel = HalfSpaceMedium(density=7850.0, longitudinal_wave_speed=5900.0, name="Steel")
        stack = LaminatedStack(layers=[layer])
        result_obj = stack.solve_sweep(freqs, left_medium=water, right_medium=steel)
        result_raw = stack.solve_sweep(freqs, left_medium_impedance=water.impedance, right_medium_impedance=steel.impedance)
        np.testing.assert_allclose(result_obj.reflection_coefficient, result_raw.reflection_coefficient)
        np.testing.assert_allclose(result_obj.input_impedance, result_raw.input_impedance)
        np.testing.assert_allclose(result_obj.power_balance, result_raw.power_balance)

    def test_frequency_response_result_properties_and_layer_field(self) -> None:
        """结果对象的幅值、相位和无耗功率平衡属性应正确。"""
        material = Material(density=1000.0, young_modulus=1.875e9, poisson_ratio=0.25, name="WaterLike")
        layer = Layer.from_material(thickness=0.9e-3, material=material)
        stack = LaminatedStack(layers=[layer])
        freqs = np.array([0.25e6, 0.75e6])
        result = stack.solve_sweep(freqs, left_medium_impedance=1.5e6, right_medium_impedance=1.5e6)
        np.testing.assert_allclose(result.reflection_magnitude, np.abs(result.reflection_coefficient))
        np.testing.assert_allclose(result.reflection_phase, np.angle(result.reflection_coefficient))
        np.testing.assert_allclose(result.input_impedance_magnitude, np.abs(result.input_impedance))
        np.testing.assert_allclose(result.power_balance, np.ones_like(result.power_balance), rtol=1e-9, atol=1e-9)

    def test_lossless_energy_balance_holds_for_real_boundary_impedances(self) -> None:
        """无耗且边界阻抗为实数时应功率守恒。"""
        aluminum = Material(density=2700.0, young_modulus=70e9, poisson_ratio=0.33, name="Aluminum")
        polymer = Material(density=1200.0, young_modulus=3e9, poisson_ratio=0.40, name="Polymer")
        composite = Material(density=1600.0, young_modulus=8e9, poisson_ratio=0.30, name="Composite")
        layers = [
            Layer.from_material(thickness=0.8e-3, material=aluminum),
            Layer.from_material(thickness=0.3e-3, material=polymer),
            Layer.from_material(thickness=1.1e-3, material=composite),
        ]
        interfaces = [InterfaceSpring(4e13), InterfaceSpring(8e12)]
        stack = LaminatedStack(layers=layers, interfaces=interfaces)
        left = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1480.0)
        right = HalfSpaceMedium(density=7850.0, longitudinal_wave_speed=5900.0)
        freqs = np.array([0.2e6, 0.8e6, 1.6e6])
        result = stack.solve_sweep(freqs, left_medium=left, right_medium=right)
        np.testing.assert_allclose(result.power_balance, np.ones_like(result.power_balance), rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
