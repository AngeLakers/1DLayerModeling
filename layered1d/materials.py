from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class Material:
    """1D effective longitudinal material definition.

    Notes
    -----
    In the current solver, only ``density`` and ``young_modulus`` participate
    in the forward model. Optional fields are carried for organization and for
    future model extensions.
    """

    density: float
    young_modulus: float
    name: str = ""
    poisson_ratio: Optional[float] = None
    attenuation_alpha: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError("density must be positive and finite.")
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive and finite.")
        if self.poisson_ratio is not None:
            if (not math.isfinite(self.poisson_ratio)) or not (-1.0 < self.poisson_ratio < 0.5):
                raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5) when provided.")
        if self.attenuation_alpha is not None:
            if not math.isfinite(self.attenuation_alpha) or self.attenuation_alpha < 0:
                raise ValueError("attenuation_alpha must be finite and non-negative when provided.")

    @property
    def wave_speed(self) -> float:
        return math.sqrt(self.young_modulus / self.density)

    @property
    def impedance(self) -> float:
        return self.density * self.wave_speed
