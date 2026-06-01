from __future__ import annotations

import math
import unittest

import numpy as np

from layered1d import HalfSpaceMedium, InterfaceSpring, LaminatedStack, Layer
from layered1d.materials import Material


def isotropic_E_from_normal_longitudinal_speed(
    rho: float,
    longitudinal_wave_speed: float,
    nu: float,
) -> float:
    return (
        rho
        * longitudinal_wave_speed
        * longitudinal_wave_speed
        * (1.0 + nu)
        * (1.0 - 2.0 * nu)
        / (1.0 - nu)
    )


def material_from_longitudinal_speed(
    rho: float,
    longitudinal_wave_speed: float,
    nu: float = 0.25,
    attenuation_alpha: float | None = None,
) -> Material:
    return Material(
        density=rho,
        young_modulus=isotropic_E_from_normal_longitudinal_speed(
            rho,
            longitudinal_wave_speed,
            nu,
        ),
        poisson_ratio=nu,
        attenuation_alpha=attenuation_alpha,
    )


def layer_transfer_matrix(layer: Layer, omega: float) -> np.ndarray:
    """手写位移-应力传递矩阵：state = [u, sigma]."""
    kh = layer.wavenumber(omega) * layer.thickness
    z = layer.impedance
    return np.array(
        [
            [np.cos(kh), np.sin(kh) / (omega * z)],
            [-omega * z * np.sin(kh), np.cos(kh)],
        ],
        dtype=complex,
    )


def spring_transfer_matrix(stiffness: float) -> np.ndarray:
    return np.array([[1.0, 1.0 / stiffness], [0.0, 1.0]], dtype=complex)


def stack_transfer_matrix(
    layers: list[Layer],
    omega: float,
    spring_stiffnesses: list[float] | None = None,
) -> np.ndarray:
    if spring_stiffnesses is None:
        spring_stiffnesses = []
    matrix = np.eye(2, dtype=complex)
    for index, layer in enumerate(layers):
        matrix = layer_transfer_matrix(layer, omega) @ matrix
        if index < len(spring_stiffnesses):
            matrix = spring_transfer_matrix(spring_stiffnesses[index]) @ matrix
    return matrix


def reflection_from_transfer_matrix(
    transfer_matrix: np.ndarray,
    omega: float,
    left_impedance: float,
    right_stress_per_displacement: complex,
) -> complex:
    m11 = transfer_matrix[0, 0]
    m12 = transfer_matrix[0, 1]
    m21 = transfer_matrix[1, 0]
    m22 = transfer_matrix[1, 1]
    y_left = (right_stress_per_displacement * m11 - m21) / (
        m22 - right_stress_per_displacement * m12
    )
    normalized_input = y_left / (-1j * omega * left_impedance)
    return (1.0 - normalized_input) / (1.0 + normalized_input)


def transfer_matrix_reflection(
    layers: list[Layer],
    frequencies_hz: np.ndarray,
    left_impedance: float,
    right_impedance: float,
    spring_stiffnesses: list[float] | None = None,
) -> np.ndarray:
    values = []
    for frequency_hz in frequencies_hz:
        omega = 2.0 * math.pi * float(frequency_hz)
        matrix = stack_transfer_matrix(layers, omega, spring_stiffnesses)
        right_load = -1j * omega * right_impedance
        values.append(
            reflection_from_transfer_matrix(
                matrix,
                omega,
                left_impedance,
                right_load,
            )
        )
    return np.asarray(values, dtype=complex)


def layer_input_impedance_from_acoustic_recurrence(
    layer: Layer,
    omega: float,
    load_impedance: complex,
) -> complex:
    """声阻抗递推公式，采用 p/v 形式的输入阻抗。"""
    phase = layer.wavenumber(omega) * layer.thickness
    layer_impedance = layer.impedance
    return layer_impedance * (
        load_impedance + 1j * layer_impedance * np.tan(phase)
    ) / (
        layer_impedance + 1j * load_impedance * np.tan(phase)
    )


def acoustic_recurrence_displacement_reflection(
    layers: list[Layer],
    frequencies_hz: np.ndarray,
    left_impedance: float,
    right_impedance: float,
) -> np.ndarray:
    """用声阻抗递推得到与当前求解器定义一致的位移反射系数。

    标准声压反射系数为 ``(Z_in - Z_0) / (Z_in + Z_0)``。当前 solver
    返回入射/反射位移幅值之比，因此需要取相反号。
    """
    values = []
    for frequency_hz in frequencies_hz:
        omega = 2.0 * math.pi * float(frequency_hz)
        input_impedance: complex = complex(right_impedance)
        for layer in reversed(layers):
            input_impedance = layer_input_impedance_from_acoustic_recurrence(
                layer,
                omega,
                input_impedance,
            )
        values.append((left_impedance - input_impedance) / (left_impedance + input_impedance))
    return np.asarray(values, dtype=complex)


def free_right_reflection(
    layer: Layer,
    frequencies_hz: np.ndarray,
    left_impedance: float,
) -> np.ndarray:
    values = []
    for frequency_hz in frequencies_hz:
        omega = 2.0 * math.pi * float(frequency_hz)
        matrix = stack_transfer_matrix([layer], omega)
        values.append(
            reflection_from_transfer_matrix(
                matrix,
                omega,
                left_impedance,
                right_stress_per_displacement=0.0,
            )
        )
    return np.asarray(values, dtype=complex)


class BenchmarkRegressionTests(unittest.TestCase):
    def make_two_layer_limit_case(self) -> tuple[list[Layer], float, float, np.ndarray]:
        first = Layer.from_material(
            thickness=0.7e-3,
            material=material_from_longitudinal_speed(1200.0, 1800.0),
        )
        second = Layer.from_material(
            thickness=0.9e-3,
            material=material_from_longitudinal_speed(2500.0, 3200.0),
        )
        frequencies_hz = np.array([0.21e6, 0.58e6, 1.13e6, 1.72e6])
        return [first, second], 1.5e6, 7.8e6, frequencies_hz

    def test_single_layer_reflection_matches_hand_transfer_matrix(self) -> None:
        """单层 R(f) 应和独立手写 transfer matrix 结果一致。"""
        material = material_from_longitudinal_speed(1800.0, 2600.0)
        layer = Layer.from_material(thickness=0.85e-3, material=material)
        left_impedance = 1.48e6
        right_impedance = 4.2e6
        frequencies_hz = np.array([0.16e6, 0.47e6, 0.94e6, 1.51e6])

        global_result = LaminatedStack([layer]).solve_sweep(
            frequencies_hz,
            left_medium_impedance=left_impedance,
            right_medium_impedance=right_impedance,
        )
        transfer_result = transfer_matrix_reflection(
            [layer],
            frequencies_hz,
            left_impedance,
            right_impedance,
        )

        np.testing.assert_allclose(
            global_result.reflection_coefficient,
            transfer_result,
            rtol=1e-11,
            atol=1e-11,
        )
        recurrence_result = acoustic_recurrence_displacement_reflection(
            [layer],
            frequencies_hz,
            left_impedance,
            right_impedance,
        )
        np.testing.assert_allclose(
            global_result.reflection_coefficient,
            recurrence_result,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_two_layer_rigid_formula_matches_acoustic_impedance_recurrence(self) -> None:
        """无界面跳跃的双层 R(f) 应和声阻抗递推公式一致。"""
        layers, left_impedance, right_impedance, frequencies_hz = self.make_two_layer_limit_case()
        transfer_result = transfer_matrix_reflection(
            layers,
            frequencies_hz,
            left_impedance,
            right_impedance,
        )
        recurrence_result = acoustic_recurrence_displacement_reflection(
            layers,
            frequencies_hz,
            left_impedance,
            right_impedance,
        )

        np.testing.assert_allclose(
            transfer_result,
            recurrence_result,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_two_layer_spring_reflection_matches_hand_transfer_matrix(self) -> None:
        """两层含有限弹簧界面的 R(f) 应和独立 transfer matrix 结果一致。"""
        layers, left_impedance, right_impedance, frequencies_hz = self.make_two_layer_limit_case()
        interface_stiffness = 7.0e12

        global_result = LaminatedStack(
            layers,
            interfaces=[InterfaceSpring(interface_stiffness)],
        ).solve_sweep(
            frequencies_hz,
            left_medium_impedance=left_impedance,
            right_medium_impedance=right_impedance,
        )
        transfer_result = transfer_matrix_reflection(
            layers,
            frequencies_hz,
            left_impedance,
            right_impedance,
            spring_stiffnesses=[interface_stiffness],
        )

        np.testing.assert_allclose(
            global_result.reflection_coefficient,
            transfer_result,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_interface_stiffness_large_limit_converges_to_rigid_connection(self) -> None:
        """K_interface -> infinity 时应收敛到无位移跳跃的刚性连接。"""
        layers, left_impedance, right_impedance, frequencies_hz = self.make_two_layer_limit_case()
        rigid_reference = acoustic_recurrence_displacement_reflection(
            layers,
            frequencies_hz,
            left_impedance,
            right_impedance,
        )
        stiffnesses = [1.0e16, 1.0e17, 1.0e18]
        errors = []
        final_result = None

        for stiffness in stiffnesses:
            result = LaminatedStack(
                layers,
                interfaces=[InterfaceSpring(stiffness)],
            ).solve_sweep(
                frequencies_hz,
                left_medium_impedance=left_impedance,
                right_medium_impedance=right_impedance,
            )
            final_result = result
            errors.append(
                np.max(np.abs(result.reflection_coefficient - rigid_reference))
            )

        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[-1], 1e-4)
        self.assertIsNotNone(final_result)
        assert final_result is not None
        self.assertLess(np.max(final_result.interface_jump_magnitude), 1e-4)

    def test_interface_stiffness_small_limit_converges_to_free_interface(self) -> None:
        """K_interface -> 0 时入射侧应收敛到第一层自由端响应。"""
        layers, left_impedance, right_impedance, frequencies_hz = self.make_two_layer_limit_case()
        free_reference = free_right_reflection(layers[0], frequencies_hz, left_impedance)
        stiffnesses = [1.0e11, 1.0e9, 1.0e7, 1.0e5]
        errors = []

        for stiffness in stiffnesses:
            result = LaminatedStack(
                layers,
                interfaces=[InterfaceSpring(stiffness)],
            ).solve_sweep(
                frequencies_hz,
                left_medium_impedance=left_impedance,
                right_medium_impedance=right_impedance,
            )
            errors.append(
                np.max(np.abs(result.reflection_coefficient - free_reference))
            )

        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[3], errors[2])
        self.assertLess(errors[-1], 1e-6)

    def phase_benchmark_response(self) -> tuple[np.ndarray, np.ndarray, Layer]:
        material = material_from_longitudinal_speed(
            1200.0,
            1800.0,
            attenuation_alpha=0.5,
        )
        layer = Layer.from_material(thickness=1.0e-3, material=material)
        water = HalfSpaceMedium(acoustic_impedance=1.5e6)
        frequencies_hz = np.linspace(0.05e6, 3.0e6, 3001)
        response = LaminatedStack([layer]).solve_sweep(
            frequencies_hz,
            left_medium=water,
            right_medium=water,
        )
        return frequencies_hz, response.reflection_coefficient, layer

    def test_unwrapped_reflection_phase_is_continuous_away_from_resonance_windows(self) -> None:
        """unwrap phase 在非共振窗口内不应出现 2*pi 跳变残留。"""
        frequencies_hz, reflection, layer = self.phase_benchmark_response()
        phase = np.unwrap(np.angle(reflection))
        phase_steps = np.abs(np.diff(phase))
        mid_frequencies = 0.5 * (frequencies_hz[1:] + frequencies_hz[:-1])
        half_wave_frequency = layer.longitudinal_wave_speed / (2.0 * layer.thickness)
        off_resonance = np.ones_like(mid_frequencies, dtype=bool)

        for harmonic in range(1, 4):
            off_resonance &= (
                np.abs(mid_frequencies - harmonic * half_wave_frequency) > 0.03e6
            )

        self.assertTrue(np.all(np.isfinite(phase)))
        self.assertLess(np.max(phase_steps[off_resonance]), 1e-2)

    def test_group_delay_peaks_track_half_wave_resonances(self) -> None:
        """group delay 主峰应落在已知半波共振/反共振附近。"""
        frequencies_hz, reflection, layer = self.phase_benchmark_response()
        phase = np.unwrap(np.angle(reflection))
        group_delay = -np.gradient(phase, 2.0 * math.pi * frequencies_hz)
        delay_magnitude = np.abs(group_delay)
        local_peak_indices = (
            np.flatnonzero(
                (delay_magnitude[1:-1] > delay_magnitude[:-2])
                & (delay_magnitude[1:-1] >= delay_magnitude[2:])
            )
            + 1
        )
        top_indices = local_peak_indices[
            np.argsort(delay_magnitude[local_peak_indices])[-3:]
        ]
        top_frequencies = np.sort(frequencies_hz[top_indices])
        expected = (
            layer.longitudinal_wave_speed
            / (2.0 * layer.thickness)
            * np.arange(1, 4)
        )

        np.testing.assert_allclose(top_frequencies, expected, rtol=0.0, atol=5.0e3)
        self.assertGreater(
            np.min(delay_magnitude[top_indices]),
            50.0 * np.median(delay_magnitude),
        )


if __name__ == "__main__":
    unittest.main()
