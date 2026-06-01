from __future__ import annotations

import unittest

from layered1d import (
    Layer,
    check_layer_as_zero_thickness_interface,
    check_zero_thickness_interface_applicability,
    classify_zero_thickness_interface_ratio,
    zero_thickness_interface_ratio,
)
from layered1d.materials import Material


def isotropic_E_from_normal_longitudinal_speed(
    rho: float,
    longitudinal_wave_speed: float,
    nu: float = 0.25,
) -> float:
    return (
        rho
        * longitudinal_wave_speed
        * longitudinal_wave_speed
        * (1.0 + nu)
        * (1.0 - 2.0 * nu)
        / (1.0 - nu)
    )


def layer_with_longitudinal_speed(thickness: float, speed: float) -> Layer:
    material = Material(
        density=1200.0,
        young_modulus=isotropic_E_from_normal_longitudinal_speed(1200.0, speed),
        poisson_ratio=0.25,
    )
    return Layer.from_material(thickness=thickness, material=material)


class ModelCheckTests(unittest.TestCase):
    def test_zero_thickness_interface_ratio_uses_h_f_over_c(self) -> None:
        """K_N 适用性指标应为 r = h_adh * f_max / c_adh。"""
        ratio = zero_thickness_interface_ratio(
            adhesive_thickness=0.1e-3,
            max_frequency_hz=1.0e6,
            adhesive_longitudinal_wave_speed=2000.0,
        )
        self.assertEqual(ratio, 0.05)

    def test_zero_thickness_interface_classification_thresholds(self) -> None:
        """r 的三段判定应和模型说明一致。"""
        self.assertEqual(
            classify_zero_thickness_interface_ratio(0.049999).category,
            "first_order",
        )
        self.assertEqual(
            classify_zero_thickness_interface_ratio(0.05).category,
            "borderline",
        )
        self.assertEqual(
            classify_zero_thickness_interface_ratio(0.099999).category,
            "borderline",
        )
        self.assertEqual(
            classify_zero_thickness_interface_ratio(0.1).category,
            "reduced_model",
        )

    def test_existing_layer_can_be_checked_as_zero_thickness_interface_candidate(self) -> None:
        """现有 Layer 可直接按厚度、波速和最高频率判定 K_N 近似适用性。"""
        layer = layer_with_longitudinal_speed(thickness=50.0e-6, speed=2000.0)
        check = check_layer_as_zero_thickness_interface(
            layer,
            max_frequency_hz=1.0e6,
        )

        self.assertAlmostEqual(check.ratio, 0.025)
        self.assertEqual(check.category, "first_order")
        self.assertTrue(check.is_first_order_applicable)

    def test_centimeter_scale_layers_are_reduced_model_when_collapsed_to_interface(self) -> None:
        """1 cm / 2 cm 实体层不应被误判为零厚度界面的一阶近似。"""
        one_cm_layer = layer_with_longitudinal_speed(thickness=0.01, speed=2000.0)
        two_cm_layer = layer_with_longitudinal_speed(thickness=0.02, speed=2000.0)

        one_cm_check = check_layer_as_zero_thickness_interface(
            one_cm_layer,
            max_frequency_hz=1.0e6,
        )
        two_cm_check = check_layer_as_zero_thickness_interface(
            two_cm_layer,
            max_frequency_hz=1.0e6,
        )

        self.assertGreaterEqual(one_cm_check.ratio, 0.1)
        self.assertGreaterEqual(two_cm_check.ratio, 0.1)
        self.assertEqual(one_cm_check.category, "reduced_model")
        self.assertEqual(two_cm_check.category, "reduced_model")

    def test_zero_thickness_interface_check_rejects_invalid_inputs(self) -> None:
        """模型检查应拒绝非物理厚度、频率和声速输入。"""
        with self.assertRaisesRegex(ValueError, "adhesive_thickness"):
            check_zero_thickness_interface_applicability(-1.0e-3, 1.0e6, 2000.0)
        with self.assertRaisesRegex(ValueError, "max_frequency_hz"):
            check_zero_thickness_interface_applicability(1.0e-3, -1.0e6, 2000.0)
        with self.assertRaisesRegex(ValueError, "adhesive_longitudinal_wave_speed"):
            check_zero_thickness_interface_applicability(1.0e-3, 1.0e6, 0.0)
        with self.assertRaisesRegex(ValueError, "ratio"):
            classify_zero_thickness_interface_ratio(-0.1)


if __name__ == "__main__":
    unittest.main()
