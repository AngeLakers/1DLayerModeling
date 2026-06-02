from __future__ import annotations

import math
import unittest

import numpy as np

from layered1d import ConstantAttenuation, HalfSpaceMedium, InterfaceSpring, LaminatedStack, Layer
from layered1d.adhesives import AdhesiveLayerPrior
from layered1d.materials import Material


class _BadAttenuationLaw:
    """测试用衰减模型：返回非法 alpha，覆盖材料参数防御分支。"""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha

    def np_per_m(self, frequency_hz: float) -> float:
        return self._alpha


class ValidationEdgeTests(unittest.TestCase):
    def test_adhesive_prior_rejects_invalid_scalar_and_range_inputs(self) -> None:
        """胶层先验应拒绝非物理标量参数和非法区间。"""
        with self.assertRaisesRegex(ValueError, "density must be positive and finite"):
            AdhesiveLayerPrior(
                name="bad",
                density=0.0,
                longitudinal_wave_speed=2000.0,
                thickness=100e-6,
            )

        with self.assertRaisesRegex(ValueError, "density_range"):
            AdhesiveLayerPrior(
                name="bad-range",
                density=1200.0,
                longitudinal_wave_speed=2000.0,
                thickness=100e-6,
                density_range=(1300.0, 1200.0),
            )

    def test_adhesive_prior_rejects_invalid_poisson_and_ratio(self) -> None:
        """先验接口应拒绝越界泊松比和负适用性比例。"""
        prior = AdhesiveLayerPrior(
            name="ok",
            density=1200.0,
            longitudinal_wave_speed=2000.0,
            thickness=100e-6,
        )
        with self.assertRaisesRegex(ValueError, "poisson_ratio"):
            prior.young_modulus(poisson_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "ratio"):
            prior.frequency_for_ratio(-0.1)

    def test_constant_attenuation_alpha_rejects_negative_omega(self) -> None:
        """常数衰减模型 alpha(omega) 应拒绝负角频率。"""
        with self.assertRaisesRegex(ValueError, "omega"):
            ConstantAttenuation(1.0).alpha(-1.0)
        self.assertEqual(ConstantAttenuation(2.5).alpha(2.0 * math.pi * 1.0e6), 2.5)

    def test_material_rejects_invalid_parameters_and_attenuation_outputs(self) -> None:
        """Material 参数和衰减输出应经过完整输入校验。"""
        with self.assertRaisesRegex(ValueError, "density"):
            Material(density=0.0, young_modulus=1.0, poisson_ratio=0.2)
        with self.assertRaisesRegex(ValueError, "young_modulus"):
            Material(density=1000.0, young_modulus=0.0, poisson_ratio=0.2)
        with self.assertRaisesRegex(ValueError, "poisson_ratio"):
            Material(density=1000.0, young_modulus=1e9, poisson_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "attenuation_alpha"):
            Material(density=1000.0, young_modulus=1e9, poisson_ratio=0.2, attenuation_alpha=-1.0)
        with self.assertRaisesRegex(TypeError, "np_per_m"):
            Material(density=1000.0, young_modulus=1e9, poisson_ratio=0.2, attenuation=object())

        material = Material(
            density=1000.0,
            young_modulus=1e9,
            poisson_ratio=0.2,
            attenuation=ConstantAttenuation(1.0),
        )
        with self.assertRaisesRegex(ValueError, "frequency_hz"):
            material.attenuation_np_per_m(-1.0)

        bad_negative = Material(
            density=1000.0,
            young_modulus=1e9,
            poisson_ratio=0.2,
            attenuation=_BadAttenuationLaw(-1.0),
        )
        with self.assertRaisesRegex(ValueError, "attenuation law returned"):
            bad_negative.attenuation_np_per_m(1.0e6)

        bad_nan = Material(
            density=1000.0,
            young_modulus=1e9,
            poisson_ratio=0.2,
            attenuation=_BadAttenuationLaw(float("nan")),
        )
        with self.assertRaisesRegex(ValueError, "attenuation law returned"):
            bad_nan.attenuation_np_per_m(1.0e6)

    def test_half_space_medium_rejects_invalid_optional_density_and_speed(self) -> None:
        """HalfSpaceMedium 的必填和可选参数都应进行有限性校验。"""
        with self.assertRaisesRegex(ValueError, "density must be positive and finite"):
            HalfSpaceMedium(density=0.0, longitudinal_wave_speed=1500.0)
        with self.assertRaisesRegex(ValueError, "longitudinal_wave_speed must be positive and finite"):
            HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=0.0)
        with self.assertRaisesRegex(ValueError, "density must be positive and finite when provided"):
            HalfSpaceMedium(acoustic_impedance=1.5e6, density=0.0)
        with self.assertRaisesRegex(ValueError, "longitudinal_wave_speed must be positive and finite when provided"):
            HalfSpaceMedium(acoustic_impedance=1.5e6, longitudinal_wave_speed=0.0)

    def test_layer_legacy_poisson_type_error_and_model_repr(self) -> None:
        """Layer 旧构造器应拒绝不可转换泊松比，并保持常用属性可访问。"""
        with self.assertRaisesRegex(ValueError, "both density and young_modulus"):
            Layer(thickness=1.0e-3, density=1000.0)
        with self.assertRaisesRegex(ValueError, "poisson_ratio must be finite"):
            Layer(thickness=1.0e-3, density=1000.0, young_modulus=1e9, poisson_ratio="not-a-number")

        material = Material(density=1200.0, young_modulus=1.5e9, poisson_ratio=0.25, name="M")
        layer = Layer.from_material(thickness=0.8e-3, material=material, name="L")
        self.assertEqual(layer.density, 1200.0)
        self.assertEqual(layer.young_modulus, 1.5e9)
        self.assertIn("Layer(thickness=", repr(layer))

    def test_stack_interfaces_length_validation_and_layer_field(self) -> None:
        """接口数量校验与 layer_field 结果形状应稳定。"""
        material = Material(density=1000.0, young_modulus=1.875e9, poisson_ratio=0.25)
        layer1 = Layer.from_material(thickness=1.0e-3, material=material)
        layer2 = Layer.from_material(thickness=0.8e-3, material=material)
        layer3 = Layer.from_material(thickness=0.6e-3, material=material)

        with self.assertRaisesRegex(ValueError, "interfaces length must equal"):
            LaminatedStack(
                layers=[layer1, layer2, layer3],
                interfaces=[InterfaceSpring(1.0e12)],
            )

        stack = LaminatedStack(layers=[layer1])
        freqs = np.array([0.2e6, 0.4e6])
        result = stack.solve_sweep(
            freqs,
            left_medium_impedance=1.5e6,
            right_medium_impedance=1.5e6,
        )
        field = result.layer_field(layer_index=0, frequency_index=0, points=11)
        self.assertEqual(field["z"].shape, (11,))
        self.assertEqual(field["u"].shape, (11,))
        self.assertTrue(np.all(np.isfinite(np.real(field["velocity"]))))
        self.assertTrue(np.all(np.isfinite(np.imag(field["velocity"]))))


if __name__ == "__main__":
    unittest.main()
