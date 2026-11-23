# lens-raytrace

![tests](https://github.com/luisrowork1992-netizen/lens-raytrace/actions/workflows/tests.yml/badge.svg)

`raytrace` is an **exact, vectorised 3D ray tracer** for rotationally symmetric optical systems: lenses, mirrors and plane plates. It traces real rays with the vector form of Snell's law, with no paraxial approximation. This lets it measure what first-order (ABCD) optics leaves out:

- spherical aberration
- caustics
- axial colour
- spot sizes

## Features

- Spherical and plane surfaces, refractive or reflective, with clear apertures
- Vector Snell's law with total-internal-reflection detection, and the law of reflection
- Materials:
  - constant index
  - Sellmeier glasses: Schott N-BK7 and fused silica (Malitson)
  - Abbe number
- Effective focal length, back focal distance and paraxial focus, all from real near-axis rays
- Longitudinal spherical aberration
- RMS spot size and best-focus search
- Helpers:
  - `singlet` lens builder
  - thick-lens `lensmaker_efl`
  - `collimated_bundle` pupil sampling
- `record=True` keeps every ray's path for drawing ray fans

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/lens-raytrace.git
cd lens-raytrace
pip install -e .
```

## Quick start

```python
import numpy as np
import raytrace as rt

lens = rt.OpticalSystem(rt.singlet(51.5, -51.5, 6.0, rt.BK7))   # mm
print(lens.efl())                     # 50.835 mm (matches the lensmaker equation)
print(lens.longitudinal_sa(12.0))     # marginal focus vs paraxial focus
print(lens.best_focus(12.0))          # (z, RMS spot radius)

pc_good = rt.OpticalSystem(rt.singlet(51.5, np.inf, 6.0, rt.BK7))
pc_bad  = rt.OpticalSystem(rt.singlet(np.inf, -51.5, 6.0, rt.BK7))
print(pc_good.longitudinal_sa(12), pc_bad.longitudinal_sa(12))   # -1.59 vs -6.38 mm
```

## Notebook

[`notebooks/ray_tracing.ipynb`](notebooks/ray_tracing.ipynb) covers five topics:

- a ray fan through a biconvex lens
- the plano-convex orientation rule
- the caustic of a spherical mirror, compared with the exact formula
- glass dispersion and axial colour
- spot diagrams at the paraxial focus and at the best focus

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- Snell's law, unit-length directions, TIR detection and the reflection law
- the lateral shift through a tilted plate
- the paraxial EFL equals the thick-lens lensmaker equation for 4 lens shapes
- the spherical-mirror axis crossing matches the exact trigonometric result to machine precision
- the plano-convex orientation rule and h² growth of spherical aberration
- best focus beats paraxial focus
- N-BK7 n_d = 1.5168 and V_d = 64.17, plus fused silica
- the axial colour of a singlet ≈ f/V
- apertures block exactly the rays outside the stop
- ray paths are reversible

## Scope and limitations

- Rotationally symmetric, sequential systems only. There are no aspheres, tilts or decentres, and no polarisation or diffraction.
- Air is treated as n = 1.

## References

- E. Hecht, *Optics*, 5th ed., Pearson (2017).
- W. J. Smith, *Modern Optical Engineering*, 4th ed., McGraw-Hill (2008).
- I. H. Malitson, *J. Opt. Soc. Am.* 55, 1205 (1965); Schott N-BK7 data sheet.

## License

MIT
