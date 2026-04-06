from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from layered1d import Layer, InterfaceSpring, LaminatedStack


def main() -> None:
    # Example: three layers with two spring interfaces.
    layers = [
        Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9, name="Al-1"),
        Layer(thickness=0.2e-3, density=1200.0, young_modulus=3.0e9, name="Polymer"),
        Layer(thickness=1.0e-3, density=2700.0, young_modulus=70e9, name="Al-2"),
    ]
    interfaces = [
        InterfaceSpring(stiffness=2.0e14, name="I1"),
        InterfaceSpring(stiffness=8.0e13, name="I2"),
    ]
    stack = LaminatedStack(layers=layers, interfaces=interfaces)

    water_impedance = 1000.0 * 1480.0
    freqs = np.linspace(0.1e6, 2.5e6, 800)
    result = stack.solve_sweep(
        freqs,
        left_medium_impedance=water_impedance,
        right_medium_impedance=water_impedance,
        incident_displacement_amplitude=1.0,
    )

    fig1 = plt.figure(figsize=(8, 4.5))
    plt.plot(result.frequencies_hz * 1e-6, result.reflection_magnitude)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|R(\omega)|$")
    plt.title("Reflection magnitude")
    plt.tight_layout()
    fig1.savefig("reflection_magnitude.png", dpi=180)

    fig2 = plt.figure(figsize=(8, 4.5))
    plt.plot(result.frequencies_hz * 1e-6, result.input_impedance_magnitude)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|Z_{in}(\omega)|$")
    plt.title("Input impedance magnitude")
    plt.tight_layout()
    fig2.savefig("input_impedance.png", dpi=180)

    fig3 = plt.figure(figsize=(8, 4.5))
    if result.interface_jumps.shape[1] > 0:
        for j in range(result.interface_jumps.shape[1]):
            plt.plot(
                result.frequencies_hz * 1e-6,
                result.interface_jump_magnitude[:, j],
                label=f"Interface {j+1}",
            )
        plt.legend()
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|\Delta u(\omega)|$")
    plt.title("Interface displacement jump magnitude")
    plt.tight_layout()
    fig3.savefig("interface_jumps.png", dpi=180)

    # Show a through-thickness field at one selected frequency.
    idx = int(np.argmax(result.reflection_magnitude))
    field = result.layer_field(layer_index=1, frequency_index=idx, points=300)
    fig4 = plt.figure(figsize=(8, 4.5))
    plt.plot(field["z"] * 1e3, np.abs(field["u"]))
    plt.xlabel("Local thickness coordinate in layer 2 (mm)")
    plt.ylabel(r"$|u(z)|$")
    plt.title(f"Field magnitude in layer 2 at {result.frequencies_hz[idx]*1e-6:.3f} MHz")
    plt.tight_layout()
    fig4.savefig("layer_field.png", dpi=180)

    print("Saved: reflection_magnitude.png, input_impedance.png, interface_jumps.png, layer_field.png")


if __name__ == "__main__":
    main()
