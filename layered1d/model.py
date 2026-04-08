from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any, Union
import math
import numpy as np

from .media import HalfSpaceMedium

MediumLike = Union[float, HalfSpaceMedium]


@dataclass(frozen=True)
class Layer:
    """1D longitudinal layer.

    Parameters
    ----------
    thickness:
        Layer thickness h [m].
    density:
        Mass density rho [kg/m^3].
    young_modulus:
        1D effective longitudinal modulus E [Pa].
    name:
        Optional label.
    """

    thickness: float
    density: float
    young_modulus: float
    name: str = ""

    @property
    def wave_speed(self) -> float:
        return math.sqrt(self.young_modulus / self.density)

    @property
    def impedance(self) -> float:
        return self.density * self.wave_speed

    def wavenumber(self, omega: float) -> complex:
        return omega / self.wave_speed

    def dynamic_stiffness(self, omega: float) -> np.ndarray:
        """Return the 2x2 dynamic stiffness matrix of the layer.

        Port convention:
        - displacement positive along +z
        - port force is the force applied *to the object* by the exterior,
          positive along +z
        """
        k = self.wavenumber(omega)
        kh = k * self.thickness
        z = self.impedance
        s = np.sin(kh)
        c = np.cos(kh)

        if abs(s) < 1e-14:
            # Avoid a hard crash at exact poles. The response is singular there;
            # the small complex perturbation regularizes the algebra numerically.
            kh = kh + 1e-12j
            s = np.sin(kh)
            c = np.cos(kh)

        cot = c / s
        csc = 1.0 / s
        return omega * z * np.array(
            [[cot, -csc], [-csc, cot]], dtype=complex
        )

    def q_from_amplitudes(self, omega: float, a_plus: complex, a_minus: complex) -> np.ndarray:
        k = self.wavenumber(omega)
        e = np.exp(1j * k * self.thickness)
        return np.array([a_plus + a_minus, a_plus * e + a_minus / e], dtype=complex)

    def amplitudes_from_boundary_displacements(self, omega: float, u_left: complex, u_right: complex) -> Tuple[complex, complex]:
        k = self.wavenumber(omega)
        e = np.exp(1j * k * self.thickness)
        d = np.array([[1.0, 1.0], [e, 1.0 / e]], dtype=complex)
        a_plus, a_minus = np.linalg.solve(d, np.array([u_left, u_right], dtype=complex))
        return a_plus, a_minus

    def field(self, omega: float, z_local: np.ndarray, u_left: complex, u_right: complex) -> Dict[str, np.ndarray]:
        """Recover displacement, stress, and particle velocity inside the layer."""
        a_plus, a_minus = self.amplitudes_from_boundary_displacements(omega, u_left, u_right)
        k = self.wavenumber(omega)
        z = self.impedance
        z_local = np.asarray(z_local, dtype=float)
        phase_p = np.exp(1j * k * z_local)
        phase_m = np.exp(-1j * k * z_local)
        u = a_plus * phase_p + a_minus * phase_m
        sigma = 1j * omega * z * (a_plus * phase_p - a_minus * phase_m)
        velocity = -1j * omega * u
        return {
            "z": z_local,
            "u": u,
            "sigma": sigma,
            "velocity": velocity,
            "a_plus": np.full_like(z_local, a_plus, dtype=complex),
            "a_minus": np.full_like(z_local, a_minus, dtype=complex),
        }


@dataclass(frozen=True)
class InterfaceSpring:
    """Zero-thickness normal spring interface with explicit finite stiffness."""

    stiffness: float
    name: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.stiffness):
            raise ValueError("Interface stiffness must be finite.")
        if self.stiffness <= 0:
            raise ValueError("Interface stiffness must be positive.")

    def dynamic_stiffness(self) -> np.ndarray:
        k = float(self.stiffness)
        return k * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=complex)


@dataclass(frozen=True)
class Connectivity:
    """Global DOF connectivity bookkeeping for layer/spring assembly."""

    num_dofs: int
    layer_dofs: List[Tuple[int, int]]
    spring_dofs: List[Tuple[int, int]]
    interface_jump_nodes: List[Tuple[int, int]]
    node_positions: np.ndarray
    node_labels: List[str]


class LaminatedStack:
    """1D laminated stack with explicit zero-thickness spring interfaces."""

    def __init__(self, layers: Sequence[Layer], interfaces: Optional[Sequence[InterfaceSpring]] = None):
        if len(layers) == 0:
            raise ValueError("At least one layer is required.")
        self.layers: List[Layer] = list(layers)
        if interfaces is None:
            interfaces = []
        if len(self.layers) > 1 and len(interfaces) == 0:
            raise ValueError("Explicit interfaces are required when len(layers) > 1.")
        if len(interfaces) != len(layers) - 1:
            raise ValueError("interfaces length must equal len(layers) - 1.")
        self.interfaces: List[InterfaceSpring] = list(interfaces)

        self._connectivity = self._build_connectivity()

    def _build_connectivity(self) -> Connectivity:
        layer_dofs: List[Tuple[int, int]] = []
        spring_dofs: List[Tuple[int, int]] = []
        node_positions: List[float] = [0.0]
        node_labels: List[str] = ["x0"]

        current_left = 0
        x = 0.0
        next_node = 1

        for j, layer in enumerate(self.layers):
            if j == len(self.layers) - 1:
                right_node = next_node
                next_node += 1
                x += layer.thickness
                node_positions.append(x)
                node_labels.append(f"x{len(node_positions)-1}")
            else:
                x += layer.thickness
                right_node = next_node
                next_node += 1
                node_positions.append(x)
                node_labels.append(f"x{len(node_positions)-1}L")
                next_left = next_node
                next_node += 1
                node_positions.append(x)
                node_labels.append(f"x{len(node_positions)-1}R")
                spring_dofs.append((right_node, next_left))
                layer_dofs.append((current_left, right_node))
                current_left = next_left
                continue

            layer_dofs.append((current_left, right_node))
            current_left = right_node

        return Connectivity(
            num_dofs=next_node,
            layer_dofs=layer_dofs,
            spring_dofs=spring_dofs,
            interface_jump_nodes=spring_dofs,
            node_positions=np.array(node_positions, dtype=float),
            node_labels=node_labels,
        )

    @property
    def num_dofs(self) -> int:
        return self._connectivity.num_dofs

    @property
    def node_positions(self) -> np.ndarray:
        return self._connectivity.node_positions.copy()

    @property
    def node_labels(self) -> List[str]:
        return list(self._connectivity.node_labels)

    def _scatter_add_2x2(self, k_global: np.ndarray, k_local: np.ndarray, i: int, j: int) -> None:
        idx = np.ix_([i, j], [i, j])
        k_global[idx] += k_local

    def assemble_structure_matrix(self, omega: float) -> np.ndarray:
        k_global = np.zeros((self.num_dofs, self.num_dofs), dtype=complex)

        for layer, (i, j) in zip(self.layers, self._connectivity.layer_dofs):
            self._scatter_add_2x2(k_global, layer.dynamic_stiffness(omega), i, j)

        for interface, dofs in zip(self.interfaces, self._connectivity.spring_dofs):
            i, j = dofs
            self._scatter_add_2x2(k_global, interface.dynamic_stiffness(), i, j)

        return k_global

    def _resolve_boundary_impedance(
        self,
        positional_value: Optional[MediumLike],
        named_value: Optional[MediumLike],
        side: str,
    ) -> float:
        provided = [value for value in (positional_value, named_value) if value is not None]
        if len(provided) != 1:
            raise ValueError(
                f"Provide exactly one of {side}_medium_impedance or {side}_medium."
            )
        value = provided[0]
        if isinstance(value, HalfSpaceMedium):
            return value.impedance
        if not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"{side} boundary impedance must be positive and finite.")
        return float(value)

    def _apply_boundary_conditions(
        self,
        k_struct: np.ndarray,
        omega: float,
        left_medium_impedance: float,
        right_medium_impedance: float,
        incident_displacement_amplitude: complex,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply semi-infinite boundary loads and left-incident excitation."""
        k_total = k_struct.copy()
        rhs = np.zeros(self.num_dofs, dtype=complex)

        k_total[0, 0] += 1j * omega * left_medium_impedance
        k_total[-1, -1] += 1j * omega * right_medium_impedance
        rhs[0] += 2j * omega * left_medium_impedance * incident_displacement_amplitude
        return k_total, rhs

    def _recover_scattering_outputs(
        self,
        omega: float,
        u: np.ndarray,
        incident_displacement_amplitude: complex,
        left_medium_impedance: float,
    ) -> Dict[str, complex]:
        u_left = u[0]
        u_right = u[-1]
        a_inc = incident_displacement_amplitude
        a_ref = u_left - a_inc
        a_tr = u_right

        velocity_left = -1j * omega * u_left
        force_left = 1j * omega * left_medium_impedance * u_left - 2j * omega * left_medium_impedance * a_inc
        input_impedance = np.inf if abs(velocity_left) == 0 else force_left / velocity_left

        return {
            "reflection_coefficient": a_ref / a_inc,
            "transmission_displacement_ratio": a_tr / a_inc,
            "input_impedance": input_impedance,
        }

    def _compute_interface_jumps(self, u: np.ndarray) -> np.ndarray:
        interface_jumps: List[complex] = []
        for dofs in self._connectivity.interface_jump_nodes:
            i, j = dofs
            interface_jumps.append(u[j] - u[i])
        return np.array(interface_jumps, dtype=complex)

    def solve_frequency_point(
        self,
        frequency_hz: float,
        left_medium_impedance: Optional[MediumLike] = None,
        right_medium_impedance: Optional[MediumLike] = None,
        incident_displacement_amplitude: complex = 1.0,
        *,
        left_medium: Optional[MediumLike] = None,
        right_medium: Optional[MediumLike] = None,
    ) -> Dict[str, Any]:
        """Solve one frequency point.

        Boundary model:
        - left medium: semi-infinite, one incoming wave + one reflected wave
        - right medium: semi-infinite, outgoing transmitted wave only

        Parameters
        ----------
        left_medium_impedance, right_medium_impedance:
            Backward-compatible scalar impedances, or ``HalfSpaceMedium`` objects.
        left_medium, right_medium:
            Preferred explicit boundary-medium objects or scalar impedances.
        """
        omega = 2.0 * math.pi * frequency_hz
        left_z = self._resolve_boundary_impedance(left_medium_impedance, left_medium, "left")
        right_z = self._resolve_boundary_impedance(right_medium_impedance, right_medium, "right")
        k_struct = self.assemble_structure_matrix(omega)
        k_total, rhs = self._apply_boundary_conditions(
            k_struct,
            omega=omega,
            left_medium_impedance=left_z,
            right_medium_impedance=right_z,
            incident_displacement_amplitude=incident_displacement_amplitude,
        )

        u = np.linalg.solve(k_total, rhs)
        outputs = self._recover_scattering_outputs(
            omega,
            u=u,
            incident_displacement_amplitude=incident_displacement_amplitude,
            left_medium_impedance=left_z,
        )

        return {
            "frequency_hz": frequency_hz,
            "omega": omega,
            "u": u,
            **outputs,
            "interface_jumps": self._compute_interface_jumps(u),
            "global_matrix": k_total,
            "rhs": rhs,
            "node_positions": self.node_positions,
            "node_labels": self.node_labels,
            "left_boundary_impedance": left_z,
            "right_boundary_impedance": right_z,
        }

    def solve_sweep(
        self,
        frequencies_hz: Sequence[float],
        left_medium_impedance: Optional[MediumLike] = None,
        right_medium_impedance: Optional[MediumLike] = None,
        incident_displacement_amplitude: complex = 1.0,
        *,
        left_medium: Optional[MediumLike] = None,
        right_medium: Optional[MediumLike] = None,
    ) -> "FrequencyResponseResult":
        from .solver import FrequencyResponseResult

        freqs = np.asarray(frequencies_hz, dtype=float)
        solutions = [
            self.solve_frequency_point(
                float(f),
                left_medium_impedance=left_medium_impedance,
                right_medium_impedance=right_medium_impedance,
                incident_displacement_amplitude=incident_displacement_amplitude,
                left_medium=left_medium,
                right_medium=right_medium,
            )
            for f in freqs
        ]
        return FrequencyResponseResult.from_solutions(self, freqs, solutions)
