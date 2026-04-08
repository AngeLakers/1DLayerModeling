from __future__ import annotations

import unittest
import numpy as np

from layered1d import HalfSpaceMedium, InterfaceSpring, LaminatedStack, Layer


class PhysicsConsistencyTests(unittest.TestCase):
    def test_dynamic_stiffness_matches_low_frequency_static_limit(self) -> None:
        """As omega->0, dynamic stiffness tends to E/h * [[1,-1],[-1,1]]."""
        layer = Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9)
        omega = 1.0

        k_dynamic = layer.dynamic_stiffness(omega)
        k_static = (layer.young_modulus / layer.thickness) * np.array(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=complex,
        )

        np.testing.assert_allclose(k_dynamic, k_static, rtol=1e-9, atol=1e-3)

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

        np.testing.assert_allclose(
            res_stiff.reflection_coefficient,
            res_stiff_ref.reflection_coefficient,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_multilayer_requires_explicit_interfaces(self) -> None:
        layers = [
            Layer(thickness=0.8e-3, density=2700.0, young_modulus=70e9),
            Layer(thickness=0.3e-3, density=1200.0, young_modulus=3e9),
        ]
        with self.assertRaises(ValueError):
            LaminatedStack(layers)


if __name__ == "__main__":
    unittest.main()
