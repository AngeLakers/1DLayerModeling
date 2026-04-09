from __future__ import annotations

import math
import unittest

import numpy as np

from layered1d import HalfSpaceMedium, InterfaceSpring, LaminatedStack, Layer
from layered1d.materials import Material


def isotropic_E_from_cl(rho: float, c_l: float, nu: float) -> float:
    return rho * c_l * c_l * (1.0 + nu) * (1.0 - 2.0 * nu) / (1.0 - nu)


class PhysicsConsistencyTests(unittest.TestCase):
    def test_material_derives_wave_speeds_and_impedance(self) -> None:
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
        with self.assertRaisesRegex(TypeError, "poisson_ratio"):
            Material(density=2700.0, young_modulus=70e9, name="Aluminum")

    def test_layer_can_be_built_from_material_or_legacy_properties(self) -> None:
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

    def test_dynamic_stiffness_matches_low_frequency_static_limit(self) -> None:
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
        rho = 1.0
        nu = 0.25
        c_l = 1.0
        E = isotropic_E_from_cl(rho, c_l, nu)
        with self.assertWarns(FutureWarning):
            layer = Layer(thickness=1.0, density=rho, young_modulus=E, poisson_ratio=nu)
        omega = math.pi * layer.longitudinal_wave_speed / layer.thickness
        k_dynamic = layer.dynamic_stiffness(omega)
        self.assertTrue(np.isfinite(k_dynamic).all())

    def test_amplitude_roundtrip_matches_boundary_displacements(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "must be finite"):
            InterfaceSpring(stiffness=float("nan"))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            InterfaceSpring(stiffness=0.0)

    def test_laminated_stack_constructor_validations(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one layer is required"):
            LaminatedStack(layers=[])

        with self.assertWarns(FutureWarning):
            layers = [
                Layer(thickness=0.5e-3, density=1000.0, young_modulus=2.0e9, poisson_ratio=0.30),
                Layer(thickness=0.7e-3, density=1200.0, young_modulus=2.5e9, poisson_ratio=0.35),
            ]
        with self.assertRaisesRegex(ValueError, "Explicit interfaces are required"):
            LaminatedStack(layers=layers, interfaces=[])

    def test_reflection_is_zero_for_impedance_matched_single_layer(self) -> None:
        rho = 1000.0
        nu = 0.25
        c_l = 1500.0
        E = isotropic_E_from_cl(rho, c_l, nu)
        material = Material(density=rho, young_modulus=E, poisson_ratio=nu, name="Matched")
        layer = Layer.from_material(thickness=1.2e-3, material=material)
        water = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1500.0, name="matched")
        stack = LaminatedStack(layers=[layer])
        freqs = np.array([0.2e6, 0.8e6, 1.4e6])
        result = stack.solve_sweep(freqs, left_medium=water, right_medium=water)
        self.assertLess(np.max(np.abs(result.reflection_coefficient)), 1e-10)

    def test_medium_object_matches_raw_impedance(self) -> None:
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
