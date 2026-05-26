from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from .attenuation import AttenuationLaw, ConstantAttenuation


@dataclass(frozen=True)
class Material:
    """Isotropic solid material for normal plane longitudinal waves.

    Notes
    -----
    In the current solver, ``longitudinal_wave_speed`` is the normal plane
    longitudinal wave speed for a laterally constrained / laterally infinite
    isotropic layer. It is derived from the longitudinal modulus ``M``:

        M = E (1 - nu) / ((1 + nu) (1 - 2 nu))
        longitudinal_wave_speed = sqrt(M / rho)

    This means ``young_modulus`` is *not* treated as the longitudinal modulus
    ``M``.

    ``attenuation_law`` defines the layer attenuation model. For backward
    compatibility, ``attenuation_alpha`` is still accepted as a shortcut for
    ``ConstantAttenuation(attenuation_alpha)``. If neither is provided, the
    layer is treated as lossless.
    """

    density: float
    young_modulus: float
    poisson_ratio: float
    name: str = ""
    attenuation_alpha: Optional[float] = None
    attenuation_law: Optional[AttenuationLaw] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError("density must be positive and finite.")
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive and finite.")
        if self.poisson_ratio is None:
            raise ValueError("poisson_ratio must be provided and finite.")
        if (not math.isfinite(self.poisson_ratio)) or not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5).")
        if self.attenuation_alpha is not None and self.attenuation_law is not None:
            raise ValueError("Provide either attenuation_alpha or attenuation_law, but not both.")
        if self.attenuation_alpha is not None:
            if not math.isfinite(self.attenuation_alpha) or self.attenuation_alpha < 0:
                raise ValueError("attenuation_alpha must be finite and non-negative when provided.")
            object.__setattr__(self, "attenuation_alpha", float(self.attenuation_alpha))
            object.__setattr__(self, "attenuation_law", ConstantAttenuation(self.attenuation_alpha))
        elif self.attenuation_law is not None:
            if not callable(getattr(self.attenuation_law, "alpha", None)):
                raise TypeError("attenuation_law must provide alpha(omega).")
            if isinstance(self.attenuation_law, ConstantAttenuation):
                object.__setattr__(self, "attenuation_alpha", self.attenuation_law.alpha_np_per_m)

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

    def attenuation_coefficient(self, omega: float) -> float:
        if self.attenuation_law is None:
            return 0.0
        alpha = self.attenuation_law.alpha(omega)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("attenuation law returned a non-finite or negative alpha.")
        return float(alpha)
