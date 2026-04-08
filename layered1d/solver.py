from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass
class FrequencyResponseResult:
    stack: Any
    frequencies_hz: np.ndarray
    nodal_displacements: np.ndarray
    reflection_coefficient: np.ndarray
    transmission_displacement_ratio: np.ndarray
    input_impedance: np.ndarray
    interface_jumps: np.ndarray
    left_boundary_impedance: np.ndarray
    right_boundary_impedance: np.ndarray
    node_positions: np.ndarray
    node_labels: List[str]
    raw_solutions: List[Dict[str, Any]]

    @classmethod
    def from_solutions(cls, stack: Any, frequencies_hz: np.ndarray, solutions: List[Dict[str, Any]]) -> "FrequencyResponseResult":
        nodal_displacements = np.vstack([s["u"] for s in solutions])
        reflection_coefficient = np.array([s["reflection_coefficient"] for s in solutions], dtype=complex)
        transmission_displacement_ratio = np.array([s["transmission_displacement_ratio"] for s in solutions], dtype=complex)
        input_impedance = np.array([s["input_impedance"] for s in solutions], dtype=complex)
        interface_jumps = np.vstack([s["interface_jumps"] for s in solutions]) if len(stack.interfaces) > 0 else np.zeros((len(frequencies_hz), 0), dtype=complex)
        left_boundary_impedance = np.array([s["left_boundary_impedance"] for s in solutions], dtype=float)
        right_boundary_impedance = np.array([s["right_boundary_impedance"] for s in solutions], dtype=float)
        return cls(
            stack=stack,
            frequencies_hz=np.asarray(frequencies_hz, dtype=float),
            nodal_displacements=nodal_displacements,
            reflection_coefficient=reflection_coefficient,
            transmission_displacement_ratio=transmission_displacement_ratio,
            input_impedance=input_impedance,
            interface_jumps=interface_jumps,
            left_boundary_impedance=left_boundary_impedance,
            right_boundary_impedance=right_boundary_impedance,
            node_positions=solutions[0]["node_positions"],
            node_labels=solutions[0]["node_labels"],
            raw_solutions=solutions,
        )

    @property
    def reflection_magnitude(self) -> np.ndarray:
        return np.abs(self.reflection_coefficient)

    @property
    def reflection_phase(self) -> np.ndarray:
        return np.angle(self.reflection_coefficient)

    @property
    def input_impedance_magnitude(self) -> np.ndarray:
        return np.abs(self.input_impedance)

    @property
    def interface_jump_magnitude(self) -> np.ndarray:
        return np.abs(self.interface_jumps)

    @property
    def power_reflectance(self) -> np.ndarray:
        return np.abs(self.reflection_coefficient) ** 2

    @property
    def power_transmittance(self) -> np.ndarray:
        return (self.right_boundary_impedance / self.left_boundary_impedance) * np.abs(self.transmission_displacement_ratio) ** 2

    @property
    def power_balance(self) -> np.ndarray:
        return self.power_reflectance + self.power_transmittance

    def layer_field(self, layer_index: int, frequency_index: int, points: int = 200) -> Dict[str, np.ndarray]:
        layer = self.stack.layers[layer_index]
        dof_left, dof_right = self.stack._connectivity.layer_dofs[layer_index]
        u_left = self.nodal_displacements[frequency_index, dof_left]
        u_right = self.nodal_displacements[frequency_index, dof_right]
        z_local = np.linspace(0.0, layer.thickness, points)
        omega = 2.0 * np.pi * self.frequencies_hz[frequency_index]
        return layer.field(omega, z_local=z_local, u_left=u_left, u_right=u_right)
