import numpy as np
import pytest

import raytrace as rt


def test_snell_law_and_unit_length():
    th = np.deg2rad(np.array([0.0, 10.0, 35.0, 60.0]))
    d = np.column_stack([np.zeros_like(th), np.sin(th), np.cos(th)])
    N = np.tile([0.0, 0.0, -1.0], (len(th), 1))
    out, tir = rt.refract(d, N, 1.0, 1.5)
    assert not tir.any()
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)
    np.testing.assert_allclose(1.5 * out[:, 1], 1.0 * np.sin(th), atol=1e-14)


def test_total_internal_reflection_is_flagged():
    th = np.deg2rad([30.0, 45.0])  # critical angle glass->air at n=1.5 is 41.8 deg
    d = np.column_stack([np.zeros(2), np.sin(th), np.cos(th)])
    _, tir = rt.refract(d, np.tile([0.0, 0.0, -1.0], (2, 1)), 1.5, 1.0)
    assert list(tir) == [False, True]


def test_reflection_law():
    d = np.array([[0.0, 0.6, 0.8]])
    out = rt.reflect(d, np.array([[0.0, 0.0, -1.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.6, -0.8]])


def test_plane_slab_lateral_shift():
    n, t, th = 1.5, 10.0, np.deg2rad(30.0)
    glass = rt.Constant(n)
    sys = rt.OpticalSystem([rt.Surface(0.0, np.inf, glass), rt.Surface(t, np.inf, rt.AIR)])
    b = sys.trace([[0.0, 0.0, -1.0]], [[0.0, np.sin(th), np.cos(th)]])
    np.testing.assert_allclose(b.dir[0], [0.0, np.sin(th), np.cos(th)], atol=1e-14)  # exits parallel
    thr = np.arcsin(np.sin(th) / n)
    y_straight = np.tan(th) * (t + 1.0)
    shift = (y_straight - b.pos[0, 1]) * np.cos(th)  # perpendicular displacement
    assert shift == pytest.approx(t * np.sin(th - thr) / np.cos(thr), rel=1e-12)


@pytest.mark.parametrize("R1, R2, t", [(50.0, -50.0, 5.0), (40.0, np.inf, 4.0), (30.0, 80.0, 3.0), (-60.0, 60.0, 2.0)])
def test_paraxial_focal_length_matches_lensmaker(R1, R2, t):
    n = 1.5
    sys = rt.OpticalSystem(rt.singlet(R1, R2, t, rt.Constant(n)))
    assert sys.efl(1e-5) == pytest.approx(rt.lensmaker_efl(R1, R2, t, n), rel=1e-7)


def test_spherical_mirror_exact_aberration():
    R = 100.0
    sys = rt.OpticalSystem([rt.Surface(0.0, -R, mirror=True)])
    assert sys.efl() == pytest.approx(R / 2, rel=1e-8)
    for h in (5.0, 20.0, 40.0):
        theta = np.arcsin(h / R)
        z_exact = -R + R / (2 * np.cos(theta))
        assert sys.axis_crossing(h) == pytest.approx(z_exact, rel=1e-12)


def test_planoconvex_orientation_rule():
    # with collimated light, the curved side should face the beam
    glass, R, t, h = rt.BK7, 51.5, 6.0, 10.0
    good = rt.OpticalSystem(rt.singlet(R, np.inf, t, glass))
    bad = rt.OpticalSystem(rt.singlet(np.inf, -R, t, glass))
    assert good.efl() == pytest.approx(bad.efl(), rel=1e-9)  # same focal length
    assert abs(bad.longitudinal_sa(h)) > 3.5 * abs(good.longitudinal_sa(h))
    assert good.longitudinal_sa(h) < 0  # positive lens: under-corrected, marginal focus is short


def test_spherical_aberration_grows_as_h_squared():
    sys = rt.OpticalSystem(rt.singlet(50.0, -50.0, 5.0, rt.Constant(1.5)))
    sa1, sa2 = sys.longitudinal_sa(1.0), sys.longitudinal_sa(2.0)
    assert sa2 / sa1 == pytest.approx(4.0, rel=2e-3)


def test_best_focus_beats_paraxial_focus():
    sys = rt.OpticalSystem(rt.singlet(50.0, -50.0, 5.0, rt.BK7))
    z_best, rms_best = sys.best_focus(8.0)
    assert rms_best < 0.5 * sys.spot_rms(sys.paraxial_focus(), 8.0)
    assert sys.axis_crossing(8.0) < z_best < sys.paraxial_focus()


def test_glass_catalogue_values():
    assert rt.BK7.n(rt.LINE_D) == pytest.approx(1.5168, abs=1e-4)
    assert rt.abbe_number(rt.BK7) == pytest.approx(64.17, abs=0.05)
    assert rt.FUSED_SILICA.n(rt.LINE_D) == pytest.approx(1.4585, abs=1e-4)
    assert rt.abbe_number(rt.FUSED_SILICA) == pytest.approx(67.8, abs=0.2)


def test_axial_colour_of_a_singlet():
    lens = rt.singlet(51.5, -51.5, 4.0, rt.BK7)
    f = {name: rt.OpticalSystem(lens, wavelength_um=wl).efl() for name, wl in
         [("F", rt.LINE_F), ("d", rt.LINE_D), ("C", rt.LINE_C)]}
    assert f["F"] < f["d"] < f["C"]
    # thin-lens estimate: f_C - f_F ~ f_d / V
    assert f["C"] - f["F"] == pytest.approx(f["d"] / rt.abbe_number(rt.BK7), rel=0.03)


def test_aperture_blocks_rays():
    sys = rt.OpticalSystem(rt.singlet(50.0, -50.0, 5.0, rt.BK7, aperture=5.5))  # stop edge between ray rings
    p, d = rt.collimated_bundle(10.0, rings=10)
    b = sys.trace(p, d)
    r = np.hypot(p[:, 0], p[:, 1])
    # a converging lens only moves rays inwards, so exactly the rays inside the stop survive
    np.testing.assert_array_equal(b.alive, r < 5.5)
    with pytest.raises(ValueError):
        sys.axis_crossing(7.0)


def test_ray_trace_is_reversible():
    lens = rt.singlet(40.0, -70.0, 6.0, rt.BK7)
    fwd = rt.OpticalSystem(lens).trace([[0.0, 3.0, -10.0]], [[0.0, 0.05, 1.0]])
    # send the outgoing ray back through the mirrored surface order
    back_surfaces = [rt.Surface(6.0, -70.0, rt.BK7), rt.Surface(0.0, 40.0, rt.AIR)]
    start = fwd.pos[0] + 20 * fwd.dir[0]
    back = rt.OpticalSystem(back_surfaces).trace([start], [-fwd.dir[0]])
    d0 = np.array([0.0, 0.05, 1.0]) / np.linalg.norm([0.0, 0.05, 1.0])
    np.testing.assert_allclose(back.dir[0], -d0, atol=1e-12)
