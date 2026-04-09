from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class Material:
    """Isotropic solid material definition for the current 1D normal-incidence model.

    Notes
    -----
    In the current solver, the longitudinal wave speed is derived under the
    isotropic-solid assumption from ``density``, ``young_modulus``, and
    ``poisson_ratio``:

        c_L = sqrt( E (1 - nu) / ( rho (1 + nu) (1 - 2 nu) ) )

    This means ``young_modulus`` is *not* treated as the longitudinal modulus
    c11. Optional fields are carried for organization and for future model
    extensions.
    """

    density: float
    young_modulus: float
    poisson_ratio: float
    name: str = ""
    attenuation_alpha: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError("density must be positive and finite.")
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive and finite.")
        if (not math.isfinite(self.poisson_ratio)) or not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5).")
        if self.attenuation_alpha is not None:
            if not math.isfinite(self.attenuation_alpha) or self.attenuation_alpha < 0:
                raise ValueError("attenuation_alpha must be finite and non-negative when provided.")

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def longitudinal_modulus(self) -> float:
        nu = self.poisson_ratio
        return self.young_modulus * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def shear_wave_speed(self) -> float:
        return math.sqrt(self.shear_modulus / self.density)

    @property
    def longitudinal_wave_speed(self) -> float:
        return math.sqrt(self.longitudinal_modulus / self.density)

    @property
    def impedance(self) -> float:
        return self.density * self.longitudinal_wave_speed
