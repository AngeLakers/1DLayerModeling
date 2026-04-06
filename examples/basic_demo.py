from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from layered1d import Layer, InterfaceSpring, LaminatedStack


def main() -> None:
    # Same three-layer structure for all samples.
    layers = [
        Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9, name="Al-1"),
        Layer(thickness=0.2e-3, density=1200.0, young_modulus=3.0e9, name="Polymer"),
        Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9, name="Al-2"),
    ]

    # Only interface stiffness changes between samples.
    sample_interfaces = {
        "Sample 1 (baseline, I2=8.0e13)": (2.0e14, 8.0e13),
        "Sample 2 (I2=2.0e12)": (2.0e14, 2.0e12),
        "Sample 3 (I2=2.0e8)": (2.0e14, 2.0e8),
    }

    water_impedance = 1000.0 * 1480.0
    f_start_hz = 0.1e6
    f_stop_hz = 2.5e6
    df_hz = 1.0e3
    # Linear frequency sampling with ~1 kHz interval (inclusive endpoints).
    freqs = np.arange(f_start_hz, f_stop_hz + df_hz, df_hz)

    # Timestamped output folder: examples/outputs/YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_results = {}
    for sample_name, (k1, k2) in sample_interfaces.items():
        interfaces = [
            InterfaceSpring(stiffness=k1, name="I1"),
            InterfaceSpring(stiffness=k2, name="I2"),
        ]
        stack = LaminatedStack(layers=layers, interfaces=interfaces)
        sample_results[sample_name] = stack.solve_sweep(
            freqs,
            left_medium_impedance=water_impedance,
            right_medium_impedance=water_impedance,
            incident_displacement_amplitude=1.0,
        )

    fig1 = plt.figure(figsize=(8, 4.5))
    for sample_name, result in sample_results.items():
        plt.plot(result.frequencies_hz * 1e-6, result.reflection_magnitude, label=sample_name)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|R(\omega)|$")
    plt.title("Reflection magnitude comparison")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(output_dir / "reflection_magnitude_comparison.png", dpi=180)

    fig2 = plt.figure(figsize=(8, 4.5))
    for sample_name, result in sample_results.items():
        plt.plot(result.frequencies_hz * 1e-6, result.input_impedance_magnitude, label=sample_name)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|Z_{in}(\omega)|$")
    plt.title("Input impedance magnitude comparison")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(output_dir / "input_impedance_comparison.png", dpi=180)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()