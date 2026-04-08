from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class HalfSpaceMedium:
    """Semi-infinite medium attached to a structural port."""

    density: Optional[float] = None
    wave_speed: Optional[float] = None
    acoustic_impedance: Optional[float] = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.acoustic_impedance is None:
            if self.density is None or self.wave_speed is None:
                raise ValueError(
                    "Provide either acoustic_impedance, or both density and wave_speed."
                )
            if not math.isfinite(self.density) or self.density <= 0:
                raise ValueError("density must be positive and finite.")
            if not math.isfinite(self.wave_speed) or self.wave_speed <= 0:
                raise ValueError("wave_speed must be positive and finite.")
        else:
            if not math.isfinite(self.acoustic_impedance) or self.acoustic_impedance <= 0:
                raise ValueError("acoustic_impedance must be positive and finite.")
            if self.density is not None and (not math.isfinite(self.density) or self.density <= 0):
                raise ValueError("density must be positive and finite when provided.")
            if self.wave_speed is not None and (not math.isfinite(self.wave_speed) or self.wave_speed <= 0):
                raise ValueError("wave_speed must be positive and finite when provided.")
            if self.density is not None and self.wave_speed is not None:
                implied = self.density * self.wave_speed
                if abs(implied - self.acoustic_impedance) > 1e-12 * max(abs(implied), abs(self.acoustic_impedance), 1.0):
                    raise ValueError(
                        "acoustic_impedance is inconsistent with density * wave_speed."
                    )

    @property
    def impedance(self) -> float:
        if self.acoustic_impedance is not None:
            return float(self.acoustic_impedance)
        assert self.density is not None and self.wave_speed is not None
        return float(self.density * self.wave_speed)

    @classmethod
    def from_impedance(cls, impedance: float, name: str = "") -> "HalfSpaceMedium":
        return cls(acoustic_impedance=impedance, name=name)
