from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math


class AttenuationLaw(Protocol):
    """Interface for layer attenuation laws."""

    def alpha(self, omega: float) -> float:
        """Return amplitude attenuation coefficient [Np/m] at angular frequency omega."""
        ...


@dataclass(frozen=True)
class ConstantAttenuation:
    """Frequency-independent amplitude attenuation coefficient [Np/m]."""

    alpha_np_per_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha_np_per_m) or self.alpha_np_per_m < 0:
            raise ValueError("alpha_np_per_m must be finite and non-negative.")
        object.__setattr__(self, "alpha_np_per_m", float(self.alpha_np_per_m))

    def alpha(self, omega: float) -> float:
        if not math.isfinite(omega):
            raise ValueError("omega must be finite.")
        return self.alpha_np_per_m
