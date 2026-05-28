from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from layered1d import (
    ConstantAttenuation,
    HalfSpaceMedium,
    InterfaceSpring,
    LaminatedStack,
    Layer,
)
from layered1d.materials import Material


def build_stack(alpha_np_per_m: float) -> LaminatedStack:
    aluminum = Material(
        density=2700.0,
        young_modulus=70e9,
        poisson_ratio=0.33,
        name="Aluminum",
    )
    polymer = Material(
        density=1200.0,
        young_modulus=3.0e9,
        poisson_ratio=0.40,
        attenuation=ConstantAttenuation(alpha_np_per_m),
        name="Polymer",
        notes="Constant attenuation is illustrative and is not a calibrated material value.",
    )
    layers = [
        Layer.from_material(thickness=1.0e-3, material=aluminum, name="Al-1"),
        Layer.from_material(thickness=0.2e-3, material=polymer, name="Polymer"),
        Layer.from_material(thickness=1.0e-3, material=aluminum, name="Al-2"),
    ]
    interfaces = [
        InterfaceSpring(stiffness=2.0e14, name="I1"),
        InterfaceSpring(stiffness=8.0e13, name="I2"),
    ]
    return LaminatedStack(layers=layers, interfaces=interfaces)


def main() -> None:
    attenuation_cases = {
        "lossless (0 Np/m)": 0.0,
        "constant low (20 Np/m)": 20.0,
        "constant high (80 Np/m)": 80.0,
    }
    left_medium = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1480.0, name="Water")
    right_medium = HalfSpaceMedium(density=7850.0, longitudinal_wave_speed=5900.0, name="Steel")
    freqs = np.arange(0.1e6, 20.0e6 + 2.5e3, 2.5e3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "outputs" / f"constant_attenuation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, alpha_np_per_m in attenuation_cases.items():
        stack = build_stack(alpha_np_per_m=alpha_np_per_m)
        results[label] = stack.solve_sweep(
            freqs,
            left_medium=left_medium,
            right_medium=right_medium,
            incident_displacement_amplitude=1.0,
        )

    fig1 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        plt.plot(result.frequencies_hz * 1e-6, result.reflection_magnitude, label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|R(\omega)|$")
    plt.title("Reflection magnitude with constant attenuation")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(output_dir / "reflection_magnitude_constant_attenuation.png", dpi=180)

    fig2 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        plt.plot(result.frequencies_hz * 1e-6, np.log10(result.input_impedance_magnitude), label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$\log_{10}(Z_{in}(\omega))$")
    plt.title("Input impedance with constant attenuation")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(output_dir / "input_impedance_constant_attenuation.png", dpi=180)

    fig3 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        max_jump = np.max(result.interface_jump_magnitude, axis=1)
        plt.plot(result.frequencies_hz * 1e-6, max_jump, label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Max interface jump magnitude")
    plt.title("Interface jump with constant attenuation")
    plt.legend()
    plt.tight_layout()
    fig3.savefig(output_dir / "interface_jump_constant_attenuation.png", dpi=180)

    fig4 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        plt.plot(result.frequencies_hz * 1e-6, result.power_balance, label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power balance")
    plt.title("Power balance with constant attenuation")
    plt.legend()
    plt.tight_layout()
    fig4.savefig(output_dir / "power_balance_constant_attenuation.png", dpi=180)

    print(f"Saved constant attenuation demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
