from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import math


ZeroThicknessInterfaceCategory = Literal[
    "first_order",
    "borderline",
    "reduced_model",
]


@dataclass(frozen=True)
class ZeroThicknessInterfaceApplicability:
    ratio: float
    category: ZeroThicknessInterfaceCategory
    recommendation: str

    @property
    def is_first_order_applicable(self) -> bool:
        return self.category == "first_order"


def zero_thickness_interface_ratio(
    adhesive_thickness: float,
    max_frequency_hz: float,
    adhesive_longitudinal_wave_speed: float,
) -> float:
    """Return r = h_adh * f_max / c_adh for zero-thickness K_N checks."""
    if not math.isfinite(adhesive_thickness) or adhesive_thickness < 0.0:
        raise ValueError("adhesive_thickness must be finite and non-negative.")
    if not math.isfinite(max_frequency_hz) or max_frequency_hz < 0.0:
        raise ValueError("max_frequency_hz must be finite and non-negative.")
    if (
        not math.isfinite(adhesive_longitudinal_wave_speed)
        or adhesive_longitudinal_wave_speed <= 0.0
    ):
        raise ValueError("adhesive_longitudinal_wave_speed must be positive and finite.")
    return adhesive_thickness * max_frequency_hz / adhesive_longitudinal_wave_speed


def classify_zero_thickness_interface_ratio(
    ratio: float,
) -> ZeroThicknessInterfaceApplicability:
    """Classify zero-thickness K_N applicability from h_adh * f_max / c_adh."""
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("ratio must be finite and non-negative.")
    if ratio < 0.05:
        return ZeroThicknessInterfaceApplicability(
            ratio=ratio,
            category="first_order",
            recommendation=(
                "zero-thickness K_N interface is usually acceptable as a first-order model"
            ),
        )
    if ratio < 0.1:
        return ZeroThicknessInterfaceApplicability(
            ratio=ratio,
            category="borderline",
            recommendation="boundary case; run sensitivity checks before relying on phase",
        )
    return ZeroThicknessInterfaceApplicability(
        ratio=ratio,
        category="reduced_model",
        recommendation=(
            "zero-thickness interface may introduce noticeable phase error; "
            "label the result as a reduced-model result"
        ),
    )


def check_zero_thickness_interface_applicability(
    adhesive_thickness: float,
    max_frequency_hz: float,
    adhesive_longitudinal_wave_speed: float,
) -> ZeroThicknessInterfaceApplicability:
    ratio = zero_thickness_interface_ratio(
        adhesive_thickness=adhesive_thickness,
        max_frequency_hz=max_frequency_hz,
        adhesive_longitudinal_wave_speed=adhesive_longitudinal_wave_speed,
    )
    return classify_zero_thickness_interface_ratio(ratio)


def check_layer_as_zero_thickness_interface(
    layer: Any,
    max_frequency_hz: float,
) -> ZeroThicknessInterfaceApplicability:
    """Apply the K_N reduced-model check to an existing finite Layer object."""
    return check_zero_thickness_interface_applicability(
        adhesive_thickness=layer.thickness,
        max_frequency_hz=max_frequency_hz,
        adhesive_longitudinal_wave_speed=layer.longitudinal_wave_speed,
    )
