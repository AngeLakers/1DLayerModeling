from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from .materials import Material
from .model import Layer
from .model_checks import (
    ZeroThicknessInterfaceApplicability,
    check_zero_thickness_interface_applicability,
)


@dataclass(frozen=True)
class AdhesiveLayerPrior:
    """Reusable literature-informed adhesive/polymer layer prior."""

    name: str
    density: float
    longitudinal_wave_speed: float
    thickness: float
    density_range: Optional[Tuple[float, float]] = None
    longitudinal_wave_speed_range: Optional[Tuple[float, float]] = None
    impedance_range: Optional[Tuple[float, float]] = None
    attenuation_alpha_range: Optional[Tuple[float, float]] = None
    source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self._require_positive_finite(self.density, "density")
        self._require_positive_finite(
            self.longitudinal_wave_speed,
            "longitudinal_wave_speed",
        )
        self._require_positive_finite(self.thickness, "thickness")
        for field_name in (
            "density_range",
            "longitudinal_wave_speed_range",
            "impedance_range",
            "attenuation_alpha_range",
        ):
            value = getattr(self, field_name)
            if value is not None:
                self._validate_range(value, field_name)

    @staticmethod
    def _require_positive_finite(value: float, name: str) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")

    @staticmethod
    def _validate_range(value: Tuple[float, float], name: str) -> None:
        lower, upper = value
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower < 0.0
            or upper < lower
        ):
            raise ValueError(f"{name} must be a finite non-negative increasing range.")

    @property
    def impedance(self) -> float:
        return self.density * self.longitudinal_wave_speed

    @property
    def longitudinal_modulus(self) -> float:
        return self.density * self.longitudinal_wave_speed**2

    def young_modulus(self, poisson_ratio: float = 0.40) -> float:
        if (
            not math.isfinite(poisson_ratio)
            or not (-1.0 < poisson_ratio < 0.5)
        ):
            raise ValueError("poisson_ratio must be finite and lie in (-1, 0.5).")
        nu = poisson_ratio
        return self.longitudinal_modulus * (1.0 + nu) * (1.0 - 2.0 * nu) / (1.0 - nu)

    def material(
        self,
        *,
        poisson_ratio: float = 0.40,
        attenuation_alpha: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Material:
        """Create a Material with the prior's rho and c_L.

        ``poisson_ratio`` is only used to convert the acoustic longitudinal
        modulus M = rho c_L^2 into the Material API's Young's modulus input.
        """
        return Material(
            density=self.density,
            young_modulus=self.young_modulus(poisson_ratio),
            poisson_ratio=poisson_ratio,
            attenuation_alpha=attenuation_alpha,
            name=name or self.name,
            notes=self.notes,
        )

    def layer(
        self,
        *,
        thickness: Optional[float] = None,
        poisson_ratio: float = 0.40,
        attenuation_alpha: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Layer:
        return Layer.from_material(
            thickness=self.thickness if thickness is None else thickness,
            material=self.material(
                poisson_ratio=poisson_ratio,
                attenuation_alpha=attenuation_alpha,
                name=name or self.name,
            ),
            name=name or self.name,
        )

    def frequency_for_ratio(
        self,
        ratio: float,
        *,
        thickness: Optional[float] = None,
        longitudinal_wave_speed: Optional[float] = None,
    ) -> float:
        if not math.isfinite(ratio) or ratio < 0.0:
            raise ValueError("ratio must be finite and non-negative.")
        h = self.thickness if thickness is None else thickness
        c = (
            self.longitudinal_wave_speed
            if longitudinal_wave_speed is None
            else longitudinal_wave_speed
        )
        self._require_positive_finite(h, "thickness")
        self._require_positive_finite(c, "longitudinal_wave_speed")
        return ratio * c / h

    def first_order_max_frequency_hz(self, *, thickness: Optional[float] = None) -> float:
        return self.frequency_for_ratio(0.05, thickness=thickness)

    def reduced_model_threshold_frequency_hz(
        self,
        *,
        thickness: Optional[float] = None,
    ) -> float:
        return self.frequency_for_ratio(0.10, thickness=thickness)

    def zero_thickness_interface_check(
        self,
        max_frequency_hz: float,
        *,
        thickness: Optional[float] = None,
        longitudinal_wave_speed: Optional[float] = None,
    ) -> ZeroThicknessInterfaceApplicability:
        return check_zero_thickness_interface_applicability(
            adhesive_thickness=self.thickness if thickness is None else thickness,
            max_frequency_hz=max_frequency_hz,
            adhesive_longitudinal_wave_speed=(
                self.longitudinal_wave_speed
                if longitudinal_wave_speed is None
                else longitudinal_wave_speed
            ),
        )


A1_DEFAULT_ADHESIVE_PRIOR = AdhesiveLayerPrior(
    name="A1 default adhesive prior",
    density=1290.0,
    longitudinal_wave_speed=2316.0,
    thickness=100.0e-6,
    density_range=(1200.0, 1400.0),
    longitudinal_wave_speed_range=(2000.0, 2600.0),
    impedance_range=(2.4e6, 3.6e6),
    attenuation_alpha_range=(0.0, 10000.0),
    source="Haldren 2019 NOA 60, Mori 2019, Ma 2024/2026 polymer references",
    notes=(
        "Literature-informed polymer/adhesive prior, not a calibrated material "
        "constant for a specific specimen."
    ),
)


NOA60_HALDREN_2019_PRIOR = AdhesiveLayerPrior(
    name="NOA 60 UV adhesive prior",
    density=1290.0,
    longitudinal_wave_speed=2316.0,
    thickness=108.3e-6,
    density_range=(1290.0, 1290.0),
    longitudinal_wave_speed_range=(1500.0, 2400.0),
    impedance_range=(1.94e6, 3.10e6),
    attenuation_alpha_range=(0.0, 10000.0),
    source="Haldren 2019",
    notes="NOA 60 UV adhesive reference; closest to a real thin adhesive layer.",
)


ADHESIVE_PRIORS = {
    "a1_default": A1_DEFAULT_ADHESIVE_PRIOR,
    "noa60_haldren_2019": NOA60_HALDREN_2019_PRIOR,
}


def make_a1_default_adhesive_material(
    *,
    poisson_ratio: float = 0.40,
    attenuation_alpha: Optional[float] = None,
) -> Material:
    return A1_DEFAULT_ADHESIVE_PRIOR.material(
        poisson_ratio=poisson_ratio,
        attenuation_alpha=attenuation_alpha,
    )


def make_a1_default_adhesive_layer(
    *,
    thickness: Optional[float] = None,
    poisson_ratio: float = 0.40,
    attenuation_alpha: Optional[float] = None,
) -> Layer:
    return A1_DEFAULT_ADHESIVE_PRIOR.layer(
        thickness=thickness,
        poisson_ratio=poisson_ratio,
        attenuation_alpha=attenuation_alpha,
    )
