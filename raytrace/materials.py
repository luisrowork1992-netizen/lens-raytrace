"""Optical materials: constant index or Sellmeier dispersion (wavelengths in micrometres)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Material", "Constant", "Sellmeier", "AIR", "VACUUM", "BK7", "FUSED_SILICA",
           "LINE_F", "LINE_D", "LINE_C", "abbe_number"]

# Fraunhofer lines (um)
LINE_F = 0.4861327   # hydrogen F
LINE_D = 0.5875618   # helium d
LINE_C = 0.6562725   # hydrogen C


class Material:
    name = "material"

    def n(self, wavelength_um: float) -> float:
        raise NotImplementedError

    def __repr__(self):
        return self.name


@dataclass(frozen=True, repr=False)
class Constant(Material):
    index: float
    name: str = "constant"

    def n(self, wavelength_um=LINE_D):
        return float(self.index)


@dataclass(frozen=True, repr=False)
class Sellmeier(Material):
    """``n^2 = 1 + sum_i B_i l^2 / (l^2 - C_i)`` with ``l`` in micrometres and C_i in um^2."""

    B: tuple
    C: tuple
    name: str = "sellmeier"

    def n(self, wavelength_um=LINE_D):
        l2 = np.asarray(wavelength_um, dtype=float) ** 2
        n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(self.B, self.C))
        return float(np.sqrt(n2)) if np.ndim(n2) == 0 else np.sqrt(n2)


def abbe_number(material: Material) -> float:
    """``V_d = (n_d - 1) / (n_F - n_C)``."""
    return (material.n(LINE_D) - 1) / (material.n(LINE_F) - material.n(LINE_C))


VACUUM = Constant(1.0, "vacuum")
AIR = Constant(1.0, "air")  # the small 0.03 % correction of real air is ignored
# Schott N-BK7 (catalogue Sellmeier coefficients)
BK7 = Sellmeier((1.03961212, 0.231792344, 1.01046945),
                (0.00600069867, 0.0200179144, 103.560653), "N-BK7")
# Fused silica, Malitson (1965)
FUSED_SILICA = Sellmeier((0.6961663, 0.4079426, 0.8974794),
                         (0.0684043**2, 0.1162414**2, 9.896161**2), "fused silica")
