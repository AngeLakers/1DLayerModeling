from __future__ import annotations

import math
import unittest

import numpy as np

from layered1d import HalfSpaceMedium, InterfaceSpring, LaminatedStack, Layer


class PhysicsConsistencyTests(unittest.TestCase):
    def test_dynamic_stiffness_matches_low_frequency_static_limit(self) -> None:
        layer = Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9)
        omega = 1.0
        k_dynamic = layer.dynamic_stiffness(omega)
        k_static = (layer.young_modulus / layer.thickness) * np.array(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=complex,
        )
        np.testing.assert_allclose(k_dynamic, k_static, rtol=1e-9, atol=1e-3)

    def test_dynamic_stiffness_regularizes_exact_sine_pole(self) -> None:
        layer = Layer(thickness=1.0, density=1.0, young_modulus=1.0)
        omega = math.pi
        k_dynamic = layer.dynamic_stiffness(omega)
        self.assertTrue(np.isfinite(k_dynamic).all())

    def test_amplitude_roundtrip_matches_boundary_displacements(self) -> None:
        layer = Layer(thickness=1.5e-3, density=1200.0, young_modulus=2.2e9)
        omega = 2.0 * math.pi * 0.7e6
        a_plus = 0.8 + 0.2j
        a_minus = -0.3 + 0.1j
        u_left, u_right = layer.q_from_amplitudes(omega, a_plus=a_plus, a_minus=a_minus)
        rec_plus, rec_minus = layer.amplitudes_from_boundary_displacements(omega, u_left=u_left, u_right=u_right)
        np.testing.assert_allclose(rec_plus, a_plus)
        np.testing.assert_allclose(rec_minus, a_minus)

    def test_field_respects_boundary_values_and_velocity_definition(self) -> None:
        layer = Layer(thickness=2.0e-3, density=1800.0, young_modulus=3.6e9)
        omega = 2.0 * math.pi * 0.3e6
        u_left = 1.2 - 0.4j
        u_right = -0.6 + 0.7j
        z_local = np.array([0.0, layer.thickness / 2.0, layer.thickness])
        field = layer.field(omega, z_local=z_local, u_left=u_left, u_right=u_right)
        np.testing.assert_allclose(field["u"][0], u_left)
        np.testing.assert_allclose(field["u"][-1], u_right)
        np.testing.assert_allclose(field["velocity"], -1j * omega * field["u"])
        np.testing.assert_allclose(field["a_plus"], np.full_like(z_local, field["a_plus"][0], dtype=complex))
        np.testing.assert_allclose(field["a_minus"], np.full_like(z_local, field["a_minus"][0], dtype=complex))

    def test_interface_spring_rejects_non_finite_or_non_positive_stiffness(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            InterfaceSpring(stiffness=float("nan"))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            InterfaceSpring(stiffness=0.0)

    def test_laminated_stack_constructor_validations(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one layer is required"):
            LaminatedStack(layers=[])

        layers = [
            Layer(thickness=0.5e-3, density=1000.0, young_modulus=2.0e9),
            Layer(thickness=0.7e-3, density=1200.0, young_modulus=2.5e9),
        ]
        with self.assertRaisesRegex(ValueError, "Explicit interfaces are required"):
            LaminatedStack(layers=layers, interfaces=[])
        with self.assertRaisesRegex(ValueError, r"interfaces length must equal len\(layers\) - 1"):
            LaminatedStack(
                layers=[*layers, Layer(thickness=0.4e-3, density=1300.0, young_modulus=2.8e9)],
                interfaces=[InterfaceSpring(1e9)],
            )
        with self.assertRaisesRegex(ValueError, r"interfaces length must equal len\(layers\) - 1"):
            LaminatedStack(layers=layers, interfaces=[InterfaceSpring(1e9), InterfaceSpring(2e9)])

    def test_reflection_is_zero_for_impedance_matched_single_layer(self) -> None:
        rho = 1000.0
        c = 1500.0
        layer = Layer(thickness=1.2e-3, density=rho, young_modulus=rho * c * c)
        water = HalfSpaceMedium(density=rho, wave_speed=c, name="matched")
        stack = LaminatedStack(layers=[layer])
        freqs = np.array([0.2e6, 0.8e6, 1.4e6])
        result = stack.solve_sweep(freqs, left_medium=water, right_medium=water)
        self.assertLess(np.max(np.abs(result.reflection_coefficient)), 1e-10)

    def test_medium_object_matches_raw_impedance(self) -> None:
        layer = Layer(thickness=0.6e-3, density=2700.0, young_modulus=70e9)
        freqs = np.array([0.4e6, 1.1e6])
        water = HalfSpaceMedium(density=1000.0, wave_speed=1480.0, name="Water")
        steel = HalfSpaceMedium(density=7850.0, wave_speed=5900.0, name="Steel")
        stack = LaminatedStack(layers=[layer])
        result_obj = stack.solve_sweep(freqs, left_medium=water, right_medium=steel)
        result_raw = stack.solve_sweep(freqs, left_medium_impedance=water.impedance, right_medium_impedance=steel.impedance)
        np.testing.assert_allclose(result_obj.reflection_coefficient, result_raw.reflection_coefficient)
        np.testing.assert_allclose(result_obj.input_impedance, result_raw.input_impedance)
        np.testing.assert_allclose(result_obj.power_balance, result_raw.power_balance)

    def test_frequency_response_result_properties_and_layer_field(self) -> None:
        layer = Layer(thickness=0.9e-3, density=1000.0, young_modulus=2.25e9)
        stack = LaminatedStack(layers=[layer])
        freqs = np.array([0.25e6, 0.75e6])
        result = stack.solve_sweep(freqs, left_medium_impedance=1.5e6, right_medium_impedance=1.5e6)
        np.testing.assert_allclose(result.reflection_magnitude, np.abs(result.reflection_coefficient))
        np.testing.assert_allclose(result.reflection_phase, np.angle(result.reflection_coefficient))
        np.testing.assert_allclose(result.input_impedance_magnitude, np.abs(result.input_impedance))
        self.assertEqual(result.interface_jumps.shape, (len(freqs), 0))
        self.assertEqual(result.interface_jump_magnitude.shape, (len(freqs), 0))
        np.testing.assert_allclose(result.power_balance, np.ones_like(result.power_balance), rtol=1e-9, atol=1e-9)
        layer_field = result.layer_field(layer_index=0, frequency_index=1, points=7)
        self.assertEqual(layer_field["z"].shape, (7,))
        np.testing.assert_allclose(layer_field["u"][0], result.nodal_displacements[1, 0])
        np.testing.assert_allclose(layer_field["u"][-1], result.nodal_displacements[1, 1])

    def test_recover_scattering_outputs_returns_infinite_impedance_for_zero_velocity(self) -> None:
        layer = Layer(thickness=1.0e-3, density=1100.0, young_modulus=2.0e9)
        stack = LaminatedStack(layers=[layer])
        outputs = stack._recover_scattering_outputs(
            omega=2.0 * math.pi * 1.0e6,
            u=np.array([0.0 + 0.0j, 0.2 + 0.1j]),
            incident_displacement_amplitude=1.0 + 0.0j,
            left_medium_impedance=1.5e6,
        )
        self.assertEqual(outputs["input_impedance"], np.inf)

    def test_lossless_energy_balance_holds_for_real_boundary_impedances(self) -> None:
        layers = [
            Layer(thickness=0.8e-3, density=2700.0, young_modulus=70e9),
            Layer(thickness=0.3e-3, density=1200.0, young_modulus=3e9),
            Layer(thickness=1.1e-3, density=1600.0, young_modulus=8e9),
        ]
        interfaces = [InterfaceSpring(4e13), InterfaceSpring(8e12)]
        stack = LaminatedStack(layers=layers, interfaces=interfaces)
        left = HalfSpaceMedium(density=1000.0, wave_speed=1480.0)
        right = HalfSpaceMedium(density=7850.0, wave_speed=5900.0)
        freqs = np.array([0.2e6, 0.8e6, 1.6e6])
        result = stack.solve_sweep(freqs, left_medium=left, right_medium=right)
        np.testing.assert_allclose(result.power_balance, np.ones_like(result.power_balance), rtol=1e-9, atol=1e-9)

    def test_transmission_power_is_reciprocal_under_reversal(self) -> None:
        layers = [
            Layer(thickness=0.7e-3, density=2700.0, young_modulus=70e9, name="A"),
            Layer(thickness=0.4e-3, density=1200.0, young_modulus=3e9, name="B"),
            Layer(thickness=1.0e-3, density=1600.0, young_modulus=8e9, name="C"),
        ]
        interfaces = [InterfaceSpring(9e13), InterfaceSpring(2e12)]
        left = HalfSpaceMedium(density=1000.0, wave_speed=1480.0, name="Water")
        right = HalfSpaceMedium(density=7850.0, wave_speed=5900.0, name="Steel")
        freqs = np.array([0.3e6, 0.9e6, 1.4e6])
        stack_lr = LaminatedStack(layers=layers, interfaces=interfaces)
        stack_rl = LaminatedStack(layers=list(reversed(layers)), interfaces=list(reversed(interfaces)))
        result_lr = stack_lr.solve_sweep(freqs, left_medium=left, right_medium=right)
        result_rl = stack_rl.solve_sweep(freqs, left_medium=right, right_medium=left)
        np.testing.assert_allclose(result_lr.power_transmittance, result_rl.power_transmittance, rtol=1e-9, atol=1e-9)

    def test_large_interface_stiffness_converges_to_rigid_reference(self) -> None:
        layers = [
            Layer(thickness=0.8e-3, density=2700.0, young_modulus=70e9),
            Layer(thickness=0.3e-3, density=1200.0, young_modulus=3e9),
        ]
        left = HalfSpaceMedium(density=1000.0, wave_speed=1480.0)
        right = HalfSpaceMedium(density=1000.0, wave_speed=1480.0)
        freqs = np.array([0.5e6, 1.0e6, 1.5e6])
        stiff_ref = LaminatedStack(layers, interfaces=[InterfaceSpring(1e24)])
        stiff = LaminatedStack(layers, interfaces=[InterfaceSpring(1e20)])
        res_stiff_ref = stiff_ref.solve_sweep(freqs, left_medium=left, right_medium=right)
        res_stiff = stiff.solve_sweep(freqs, left_medium=left, right_medium=right)
        np.testing.assert_allclose(res_stiff.reflection_coefficient, res_stiff_ref.reflection_coefficient, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
