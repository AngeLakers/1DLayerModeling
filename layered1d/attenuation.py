from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math


class AttenuationLaw(Protocol):
    """Interface for layer attenuation laws."""

    def np_per_m(self, frequency_hz: float) -> float:
        """Return amplitude attenuation coefficient [Np/m] at frequency_hz."""
        ...


@dataclass(frozen=True)
class ConstantAttenuation:
    """Frequency-independent amplitude attenuation coefficient [Np/m]."""

    alpha_np_per_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_np_per_m) or self.alpha_np_per_m < 0:
            raise ValueError("alpha_np_per_m must be finite and non-negative.")
        object.__setattr__(self, "alpha_np_per_m", float(self.alpha_np_per_m))

    def np_per_m(self, frequency_hz: float) -> float:
        _validate_frequency_hz(frequency_hz)
        return self.alpha_np_per_m

    def alpha(self, omega: float) -> float:
        if not math.isfinite(omega) or omega < 0:
            raise ValueError("omega must be finite and non-negative.")
        return self.np_per_m(omega / (2.0 * math.pi))


@dataclass(frozen=True)
class PowerLawAttenuation:
    """Frequency-dependent amplitude attenuation with a power-law frequency trend."""

    alpha_ref: float
    ref_frequency_hz: float = 20e6
    power: float = 1.0
    unit: str = "Np/m"

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_ref) or self.alpha_ref < 0:
            raise ValueError("alpha_ref must be finite and non-negative.")
        if not math.isfinite(self.ref_frequency_hz) or self.ref_frequency_hz <= 0:
            raise ValueError("ref_frequency_hz must be positive and finite.")
        if not math.isfinite(self.power) or self.power < 0:
            raise ValueError("power must be finite and non-negative.")
        if self.unit not in ("Np/m", "dB/mm"):
            raise ValueError('unit must be either "Np/m" or "dB/mm".')
        object.__setattr__(self, "alpha_ref", float(self.alpha_ref))
        object.__setattr__(self, "ref_frequency_hz", float(self.ref_frequency_hz))
        object.__setattr__(self, "power", float(self.power))

    def np_per_m(self, frequency_hz: float) -> float:
        frequency_hz = _validate_frequency_hz(frequency_hz)
        alpha_ref_np_per_m = self.alpha_ref
        if self.unit == "dB/mm":
            alpha_ref_np_per_m = self.alpha_ref * math.log(10.0) / 20.0 * 1000.0
        return alpha_ref_np_per_m * (frequency_hz / self.ref_frequency_hz) ** self.power


def _validate_frequency_hz(frequency_hz: float) -> float:
    if not math.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError("frequency_hz must be finite and non-negative.")
    return float(frequency_hz)
