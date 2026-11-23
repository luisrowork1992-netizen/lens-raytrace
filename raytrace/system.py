"""Vectorised, exact (non-paraxial) ray tracing through rotationally symmetric systems.

The optical axis is ``z``. Each :class:`Surface` sits at a vertex position ``z``
with a signed radius of curvature ``R`` (positive when the centre of curvature
lies at larger ``z``; ``inf`` for a plane). ``material`` is the medium *after*
the surface. Mirrors reflect, and rays then travel towards smaller ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import AIR, LINE_D, Material

__all__ = ["Surface", "RayBundle", "OpticalSystem", "refract", "reflect",
           "collimated_bundle", "singlet", "lensmaker_efl"]

_EPS = 1e-12


def refract(d, N, n1, n2):
    """Vector form of Snell's law.

    ``d`` are unit directions (k, 3); ``N`` unit normals facing the incoming ray
    (``d . N < 0``). Returns ``(new_directions, tir_mask)``.
    """
    r = n1 / n2
    cosi = -np.einsum("ij,ij->i", d, N)
    sin2t = r * r * (1.0 - cosi * cosi)
    tir = sin2t > 1.0
    cost = np.sqrt(np.clip(1.0 - sin2t, 0.0, None))
    out = r * d + (r * cosi - cost)[:, None] * N
    return out / np.linalg.norm(out, axis=1, keepdims=True), tir


def reflect(d, N):
    return d - 2.0 * np.einsum("ij,ij->i", d, N)[:, None] * N


@dataclass
class Surface:
    z: float
    R: float = np.inf
    material: Material = AIR
    aperture: float = np.inf      # clear semi-diameter
    mirror: bool = False

    def intersect(self, p, d):
        """Distance along each ray to the surface (nan where it misses)."""
        if np.isinf(self.R):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (self.z - p[:, 2]) / d[:, 2]
            return np.where(t > _EPS, t, np.nan)
        c = np.array([0.0, 0.0, self.z + self.R])
        oc = p - c
        b = np.einsum("ij,ij->i", d, oc)
        cc = np.einsum("ij,ij->i", oc, oc) - self.R**2
        disc = b * b - cc
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t1, t2 = -b - sq, -b + sq
        # keep the root on the cap that contains the vertex
        z1 = p[:, 2] + t1 * d[:, 2]
        on_cap1 = (z1 - c[2]) * np.sign(self.R) < 0
        t = np.where(on_cap1, t1, t2)
        return np.where(t > _EPS, t, np.nan)

    def normal(self, h):
        if np.isinf(self.R):
            N = np.zeros_like(h)
            N[:, 2] = 1.0
            return N
        c = np.array([0.0, 0.0, self.z + self.R])
        N = h - c
        return N / np.linalg.norm(N, axis=1, keepdims=True)


@dataclass
class RayBundle:
    pos: np.ndarray        # (k, 3)
    dir: np.ndarray        # (k, 3) unit vectors
    alive: np.ndarray      # (k,) bool
    path: list = field(default_factory=list)   # positions at each surface (for plotting)

    def at_z(self, z):
        """Ray positions where they cross the plane ``z`` (x, y)."""
        t = (z - self.pos[:, 2]) / self.dir[:, 2]
        return (self.pos + t[:, None] * self.dir)[:, :2]


class OpticalSystem:
    def __init__(self, surfaces, wavelength_um: float = LINE_D, object_medium: Material = AIR):
        self.surfaces = list(surfaces)
        self.wavelength = wavelength_um
        self.object_medium = object_medium

    def trace(self, pos, dir, record: bool = False) -> RayBundle:
        p = np.atleast_2d(np.asarray(pos, dtype=float)).copy()
        d = np.atleast_2d(np.asarray(dir, dtype=float)).copy()
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        alive = np.ones(len(p), dtype=bool)
        n1 = self.object_medium.n(self.wavelength)
        path = [p.copy()] if record else []
        for s in self.surfaces:
            t = s.intersect(p, d)
            hit = alive & np.isfinite(t)
            t = np.where(hit, t, 0.0)
            p = p + t[:, None] * d
            hit &= (p[:, 0] ** 2 + p[:, 1] ** 2) <= s.aperture**2
            N = s.normal(p)
            N = np.where((np.einsum("ij,ij->i", d, N) > 0)[:, None], -N, N)
            if s.mirror:
                d = np.where(hit[:, None], reflect(d, N), d)
                n2 = n1
            else:
                n2 = s.material.n(self.wavelength)
                new, tir = refract(d, N, n1, n2)
                hit &= ~tir
                d = np.where(hit[:, None], new, d)
            alive = hit
            n1 = n2
            if record:
                path.append(p.copy())
        return RayBundle(p, d, alive, path)

    # --- first-order properties ------------------------------------------
    def entry_z(self) -> float:
        """A plane safely in front of every surface (accounts for surface sag)."""
        return min(s.z - (abs(s.R) if np.isfinite(s.R) else 0.0) for s in self.surfaces) - 1.0

    def _parallel_ray(self, h):
        return self.trace([[0.0, h, self.entry_z()]], [[0.0, 0.0, 1.0]])

    def axis_crossing(self, h: float) -> float:
        """Where a ray entering parallel to the axis at height ``h`` crosses the axis."""
        b = self._parallel_ray(h)
        if not b.alive[0]:
            raise ValueError(f"ray at height {h} is blocked")
        (x, y, z), (dx, dy, dz) = b.pos[0], b.dir[0]
        return float(z - y * dz / dy)

    def efl(self, h: float = 1e-4) -> float:
        """Effective focal length from a near-axis ray (positive for converging systems)."""
        b = self._parallel_ray(h)
        dy, dz = b.dir[0, 1], b.dir[0, 2]
        return float(-h * abs(dz) / dy)

    def paraxial_focus(self) -> float:
        return self.axis_crossing(1e-4)

    def back_focal_distance(self) -> float:
        return self.paraxial_focus() - self.surfaces[-1].z

    def longitudinal_sa(self, h: float) -> float:
        """Longitudinal spherical aberration: marginal minus paraxial focus position."""
        return self.axis_crossing(h) - self.paraxial_focus()

    def spot_rms(self, z: float, radius: float, rings: int = 12) -> float:
        """RMS spot radius at plane ``z`` for a collimated on-axis beam of the given radius."""
        p, d = collimated_bundle(radius, rings, z0=self.entry_z())
        b = self.trace(p, d)
        xy = b.at_z(z)[b.alive]
        return float(np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1))))

    def best_focus(self, radius: float, rings: int = 12):
        """Plane of minimum RMS spot (searched around the paraxial focus). Returns (z, rms)."""
        from scipy.optimize import minimize_scalar
        f0 = self.paraxial_focus()
        span = max(abs(self.longitudinal_sa(radius)), 1e-6) * 2
        res = minimize_scalar(lambda z: self.spot_rms(z, radius, rings),
                              bounds=(f0 - span, f0 + span), method="bounded",
                              options={"xatol": 1e-9 * max(1, abs(f0))})
        return float(res.x), float(res.fun)


def collimated_bundle(radius: float, rings: int = 10, z0: float = -1.0, angle_rad: float = 0.0):
    """Polar grid of parallel rays filling a circular pupil (equal area per ray)."""
    pts = [(0.0, 0.0)]
    for k in range(1, rings + 1):
        r = radius * k / rings
        m = 6 * k
        th = 2 * np.pi * np.arange(m) / m
        pts += list(zip(r * np.cos(th), r * np.sin(th)))
    pts = np.array(pts)
    pos = np.column_stack([pts, np.full(len(pts), z0)])
    d = np.tile([0.0, np.sin(angle_rad), np.cos(angle_rad)], (len(pts), 1))
    return pos, d


def singlet(R1, R2, thickness, glass: Material, z0: float = 0.0, aperture: float = np.inf):
    """Two surfaces of a single lens in air, first vertex at ``z0``."""
    return [Surface(z0, R1, glass, aperture), Surface(z0 + thickness, R2, AIR, aperture)]


def lensmaker_efl(R1, R2, thickness, n):
    """Thick-lens lensmaker's equation."""
    inv = lambda R: 0.0 if np.isinf(R) else 1.0 / R
    power = (n - 1) * (inv(R1) - inv(R2) + (n - 1) * thickness * inv(R1) * inv(R2) / n)
    return 1.0 / power
