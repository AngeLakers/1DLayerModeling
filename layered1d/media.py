from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import warnings


@dataclass(frozen=True, init=False)
class HalfSpaceMedium:
    """Semi-infinite medium attached to a structural port.

    ``longitudinal_wave_speed`` means longitudinal wave speed (P-wave speed), in m/s.
    """

    density: Optional[float] = None
    longitudinal_wave_speed: Optional[float] = None  # Longitudinal wave speed (P-wave speed), m/s.
    acoustic_impedance: Optional[float] = None
    name: str = ""

    def __init__(
        self,
        density: Optional[float] = None,
        longitudinal_wave_speed: Optional[float] = None,
        acoustic_impedance: Optional[float] = None,
        name: str = "",
        *,
        wave_speed: Optional[float] = None,
    ) -> None:
        if wave_speed is not None:
            if longitudinal_wave_speed is not None and not math.isclose(
                float(wave_speed),
                float(longitudinal_wave_speed),
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError("Provide only one of wave_speed or longitudinal_wave_speed.")
            warnings.warn(
                "HalfSpaceMedium(..., wave_speed=...) is supported for compatibility; "
                "prefer longitudinal_wave_speed=...",
                FutureWarning,
                stacklevel=2,
            )
            longitudinal_wave_speed = float(wave_speed)

        object.__setattr__(self, "density", density)
        object.__setattr__(self, "longitudinal_wave_speed", longitudinal_wave_speed)
        object.__setattr__(self, "acoustic_impedance", acoustic_impedance)
        object.__setattr__(self, "name", name)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.acoustic_impedance is None:
            if self.density is None or self.longitudinal_wave_speed is None:
                raise ValueError(
                    "Provide either acoustic_impedance, or both density and longitudinal_wave_speed."
                )
            if not math.isfinite(self.density) or self.density <= 0:
                raise ValueError("density must be positive and finite.")
            if not math.isfinite(self.longitudinal_wave_speed) or self.longitudinal_wave_speed <= 0:
                raise ValueError("longitudinal_wave_speed must be positive and finite.")
        else:
            if not math.isfinite(self.acoustic_impedance) or self.acoustic_impedance <= 0:
                raise ValueError("acoustic_impedance must be positive and finite.")
            if self.density is not None and (not math.isfinite(self.density) or self.density <= 0):
                raise ValueError("density must be positive and finite when provided.")
            if self.longitudinal_wave_speed is not None and (not math.isfinite(self.longitudinal_wave_speed) or self.longitudinal_wave_speed <= 0):
                raise ValueError("longitudinal_wave_speed must be positive and finite when provided.")
            if self.density is not None and self.longitudinal_wave_speed is not None:
                implied = self.density * self.longitudinal_wave_speed
                if abs(implied - self.acoustic_impedance) > 1e-12 * max(abs(implied), abs(self.acoustic_impedance), 1.0):
                    raise ValueError(
                        "acoustic_impedance is inconsistent with density * longitudinal_wave_speed."
                    )

    @property
    def impedance(self) -> float:
        if self.acoustic_impedance is not None:
            return float(self.acoustic_impedance)
        assert self.density is not None and self.longitudinal_wave_speed is not None
        return float(self.density * self.longitudinal_wave_speed)

    @property
    def wave_speed(self) -> Optional[float]:
        # Backward-compatible alias. Prefer longitudinal_wave_speed.
        return self.longitudinal_wave_speed

    @classmethod
    def from_impedance(cls, impedance: float, name: str = "") -> "HalfSpaceMedium":
        return cls(acoustic_impedance=impedance, name=name)
