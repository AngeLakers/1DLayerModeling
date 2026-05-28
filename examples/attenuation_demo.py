from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from layered1d import (
    AttenuationLaw,
    ConstantAttenuation,
    HalfSpaceMedium,
    InterfaceSpring,
    LaminatedStack,
    Layer,
    PowerLawAttenuation,
)
from layered1d.materials import Material


def build_stack(polymer_attenuation: AttenuationLaw) -> LaminatedStack:
    aluminum = Material(
        density=2700.0,
        young_modulus=70e9,
        poisson_ratio=0.33,
        attenuation=ConstantAttenuation(0.0),
        name="Aluminum",
    )
    polymer = Material(
        density=1200.0,
        young_modulus=3.0e9,
        poisson_ratio=0.40,
        attenuation=polymer_attenuation,
        name="Polymer",
        notes="attenuation is illustrative and is not a calibrated material value.",
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
        "lossless (0 Np/m)": ConstantAttenuation(0.0),
        "constant (20 Np/m)": ConstantAttenuation(20.0),
        "power law (0.10 dB/mm @ 20 MHz)": PowerLawAttenuation(
            alpha_ref=0.10,
            ref_frequency_hz=20e6,
            power=1.0,
            unit="dB/mm",
        ),
    }
    left_medium = HalfSpaceMedium(density=1000.0, longitudinal_wave_speed=1480.0, name="Water")
    right_medium = HalfSpaceMedium(density=7850.0, longitudinal_wave_speed=5900.0, name="Steel")
    freqs = np.arange(0.1e6, 20.0e6 + 2.5e3, 2.5e3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "outputs" / f"attenuation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, attenuation in attenuation_cases.items():
        stack = build_stack(polymer_attenuation=attenuation)
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
    plt.title("Reflection magnitude vs. layer attenuation")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(output_dir / "reflection_magnitude_attenuation_comparison.png", dpi=180)

    fig2 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        plt.plot(result.frequencies_hz * 1e-6, np.log10(result.input_impedance_magnitude), label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$\log_{10}(Z_{in}(\omega))$")
    plt.title("Input impedance vs. layer attenuation")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(output_dir / "input_impedance_attenuation_comparison.png", dpi=180)

    fig3 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        max_jump = np.max(result.interface_jump_magnitude, axis=1)
        plt.plot(result.frequencies_hz * 1e-6, max_jump, label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Max interface jump magnitude")
    plt.title("Interface jump response vs. layer attenuation")
    plt.legend()
    plt.tight_layout()
    fig3.savefig(output_dir / "interface_jump_attenuation_comparison.png", dpi=180)

    fig4 = plt.figure(figsize=(8, 4.5))
    for label, result in results.items():
        plt.plot(result.frequencies_hz * 1e-6, result.power_balance, label=label)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power balance")
    plt.title("Power balance vs. layer attenuation")
    plt.legend()
    plt.tight_layout()
    fig4.savefig(output_dir / "power_balance_attenuation_comparison.png", dpi=180)

    print(f"Saved attenuation demo outputs to: {output_dir}")


if __name__ == "__main__":
    main()
