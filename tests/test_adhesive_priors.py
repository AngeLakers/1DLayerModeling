from __future__ import annotations

import unittest

from layered1d import (
    A1_DEFAULT_ADHESIVE_PRIOR,
    ADHESIVE_PRIORS,
    NOA60_HALDREN_2019_PRIOR,
    make_a1_default_adhesive_layer,
    make_a1_default_adhesive_material,
)


class AdhesivePriorTests(unittest.TestCase):
    def test_a1_default_adhesive_prior_matches_literature_informed_values(self) -> None:
        """A1 默认胶层先验应集中维护 rho、c_L、Z、h 和 alpha 敏感性范围。"""
        prior = A1_DEFAULT_ADHESIVE_PRIOR

        self.assertEqual(prior.thickness, 100.0e-6)
        self.assertEqual(prior.density, 1290.0)
        self.assertEqual(prior.longitudinal_wave_speed, 2316.0)
        self.assertAlmostEqual(prior.impedance, 2.98764e6)
        self.assertEqual(prior.density_range, (1200.0, 1400.0))
        self.assertEqual(prior.longitudinal_wave_speed_range, (2000.0, 2600.0))
        self.assertEqual(prior.impedance_range, (2.4e6, 3.6e6))
        self.assertEqual(prior.attenuation_alpha_range, (0.0, 10000.0))
        self.assertIn("a1_default", ADHESIVE_PRIORS)

    def test_a1_default_prior_builds_reusable_material_and_layer(self) -> None:
        """默认胶层先验应能复用生成 Material 和有限厚度 Layer。"""
        material = make_a1_default_adhesive_material()
        layer = make_a1_default_adhesive_layer()

        self.assertEqual(layer.thickness, 100.0e-6)
        self.assertEqual(material.density, A1_DEFAULT_ADHESIVE_PRIOR.density)
        self.assertAlmostEqual(
            material.longitudinal_wave_speed,
            A1_DEFAULT_ADHESIVE_PRIOR.longitudinal_wave_speed,
        )
        self.assertAlmostEqual(
            layer.longitudinal_wave_speed,
            A1_DEFAULT_ADHESIVE_PRIOR.longitudinal_wave_speed,
        )
        self.assertIn("Literature-informed", material.notes)

    def test_a1_default_prior_frequency_ranges_for_zero_thickness_kn(self) -> None:
        """100 um 胶层折叠为 K_N 时，应给出可复用的频率边界。"""
        prior = A1_DEFAULT_ADHESIVE_PRIOR

        self.assertAlmostEqual(prior.first_order_max_frequency_hz(), 1.158e6)
        self.assertAlmostEqual(prior.reduced_model_threshold_frequency_hz(), 2.316e6)
        self.assertAlmostEqual(
            prior.frequency_for_ratio(
                0.05,
                longitudinal_wave_speed=2000.0,
            ),
            1.0e6,
        )
        self.assertAlmostEqual(
            prior.frequency_for_ratio(
                0.05,
                longitudinal_wave_speed=2600.0,
            ),
            1.3e6,
        )

    def test_a1_default_prior_is_reduced_model_if_collapsed_to_kn_at_20_mhz(self) -> None:
        """A1 100 um 胶层若折叠为零厚度 K_N，20 MHz 下应标注 reduced-model。"""
        check = A1_DEFAULT_ADHESIVE_PRIOR.zero_thickness_interface_check(20.0e6)

        self.assertAlmostEqual(check.ratio, 20.0e6 * 100.0e-6 / 2316.0)
        self.assertEqual(check.category, "reduced_model")

    def test_haldren_noa60_reference_remains_available(self) -> None:
        """Haldren NOA 60 参考先验应独立保留，避免和 A1 默认厚度混淆。"""
        self.assertEqual(NOA60_HALDREN_2019_PRIOR.thickness, 108.3e-6)
        self.assertEqual(NOA60_HALDREN_2019_PRIOR.density, 1290.0)
        self.assertEqual(NOA60_HALDREN_2019_PRIOR.longitudinal_wave_speed, 2316.0)
        self.assertIn("noa60_haldren_2019", ADHESIVE_PRIORS)


if __name__ == "__main__":
    unittest.main()
