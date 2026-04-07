from __future__ import annotations

import unittest
import numpy as np

from layered1d import InterfaceSpring, LaminatedStack, Layer


class PhysicsConsistencyTests(unittest.TestCase):
    def test_dynamic_stiffness_matches_low_frequency_static_limit(self) -> None:
        """As omega->0, dynamic stiffness tends to E/h * [[1,-1],[-1,1]]."""
        layer = Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9)
        omega = 1.0  # rad/s, sufficiently low for kh << 1

        k_dynamic = layer.dynamic_stiffness(omega)
        k_static = (layer.young_modulus / layer.thickness) * np.array(
            [[1.0, -1.0], [-1.0, 1.0]],
            dtype=complex,
        )

        np.testing.assert_allclose(k_dynamic, k_static, rtol=1e-9, atol=1e-3)

    def test_reflection_is_zero_for_impedance_matched_single_layer(self) -> None:
        """If media and layer impedances match, reflection should be ~0."""
        rho = 1000.0
        c = 1500.0
        layer = Layer(thickness=1.2e-3, density=rho, young_modulus=rho * c * c)
        z_match = rho * c

        stack = LaminatedStack(layers=[layer])
        freqs = np.array([0.2e6, 0.8e6, 1.4e6])
        result = stack.solve_sweep(
            freqs,
            left_medium_impedance=z_match,
            right_medium_impedance=z_match,
            incident_displacement_amplitude=1.0,
        )

        self.assertLess(np.max(np.abs(result.reflection_coefficient)), 1e-10)

    def test_large_interface_stiffness_converges_to_perfect_interface(self) -> None:
        """Finite very-large spring stiffness should approach perfect bonding."""
        layers = [
            Layer(thickness=0.8e-3, density=2700.0, young_modulus=70e9),
            Layer(thickness=0.3e-3, density=1200.0, young_modulus=3e9),
        ]
        z_med = 1000.0 * 1480.0
        freqs = np.array([0.5e6, 1.0e6, 1.5e6])

        perfect = LaminatedStack(layers, interfaces=[InterfaceSpring(None)])
        stiff = LaminatedStack(layers, interfaces=[InterfaceSpring(1e20)])

        res_perfect = perfect.solve_sweep(freqs, z_med, z_med)
        res_stiff = stiff.solve_sweep(freqs, z_med, z_med)

        np.testing.assert_allclose(
            res_stiff.reflection_coefficient,
            res_perfect.reflection_coefficient,
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
