from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import warnings

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

    ``attenuation`` defines the layer attenuation model. For backward
    compatibility, ``attenuation_alpha`` is still accepted as a shortcut for
    ``ConstantAttenuation(attenuation_alpha)``, and ``attenuation_law`` remains
    as a deprecated alias. If none is provided, the layer is treated as lossless.
    """

    density: float
    young_modulus: float
    poisson_ratio: float
    name: str = ""
    attenuation_alpha: Optional[float] = None
    attenuation_law: Optional[AttenuationLaw] = None
    notes: str = ""
    attenuation: Optional[AttenuationLaw] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError("density must be positive and finite.")
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive and finite.")
        if self.poisson_ratio is None:
            raise ValueError("poisson_ratio must be provided and finite.")
        if (not math.isfinite(self.poisson_ratio)) or not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5).")

        provided_attenuation_fields = sum(
            value is not None
            for value in (self.attenuation, self.attenuation_alpha, self.attenuation_law)
        )
        if provided_attenuation_fields > 1:
            raise ValueError("Provide at most one of attenuation, attenuation_alpha, or attenuation_law.")

        normalized_attenuation = self.attenuation
        if self.attenuation_alpha is not None:
            if not math.isfinite(self.attenuation_alpha) or self.attenuation_alpha < 0:
                raise ValueError("attenuation_alpha must be finite and non-negative when provided.")
            object.__setattr__(self, "attenuation_alpha", float(self.attenuation_alpha))
            normalized_attenuation = ConstantAttenuation(self.attenuation_alpha)
        elif self.attenuation_law is not None:
            warnings.warn(
                "attenuation_law is deprecated; use attenuation instead.",
                FutureWarning,
                stacklevel=2,
            )
            normalized_attenuation = self.attenuation_law

        if normalized_attenuation is not None:
            if not callable(getattr(normalized_attenuation, "np_per_m", None)):
                raise TypeError("attenuation must provide np_per_m(frequency_hz).")
            object.__setattr__(self, "attenuation", normalized_attenuation)
            object.__setattr__(self, "attenuation_law", normalized_attenuation)
            if isinstance(normalized_attenuation, ConstantAttenuation):
                object.__setattr__(self, "attenuation_alpha", normalized_attenuation.alpha_np_per_m)

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

    def attenuation_np_per_m(self, frequency_hz: float) -> float:
        if not math.isfinite(frequency_hz) or frequency_hz < 0:
            raise ValueError("frequency_hz must be finite and non-negative.")
        if self.attenuation is None:
            return 0.0
        alpha = self.attenuation.np_per_m(frequency_hz)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("attenuation law returned a non-finite or negative alpha.")
        return float(alpha)

    def attenuation_coefficient(self, omega: float) -> float:
        if not math.isfinite(omega) or omega < 0:
            raise ValueError("omega must be finite and non-negative.")
        return self.attenuation_np_per_m(omega / (2.0 * math.pi))
