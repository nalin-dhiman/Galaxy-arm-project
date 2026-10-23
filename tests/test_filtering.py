"""Scientific invariants and failure cases for masking, filters and registration."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from galaxy_lab.filtering import (ScienceImage, DiskGeometry, annulus_mask,
    robust_background, filter_bank, display_grayscale, display_transform,
    evaluate_score, foreground_point_mask, match_gaussian_psf, reproject_to,
    inverse_variance_coadd, refine_outer_geometry)


def geometry(shape=(160, 160), scale=.4):
    return DiskGeometry(((shape[1] - 1) / 2, (shape[0] - 1) / 2), 20, 30, 8, 23)


def wcs_for(shape=(160, 160), scale=.4, shift=(0, 0)):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [(shape[1] + 1) / 2 + shift[0], (shape[0] + 1) / 2 + shift[1]]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    w.wcs.crval = [150, 20]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_masked_huge_values_do_not_leak_into_any_filter():
    rng = np.random.default_rng(3)
    data = rng.normal(12, .05, (160, 160))
    mask = np.zeros_like(data, bool)
    mask[70:73, 90:93] = True
    a = data.copy(); b = data.copy()
    a[mask] = 1e12; b[mask] = np.nan
    one = filter_bank(ScienceImage(a, .4, mask=mask), geometry(), (.4, .8, 1.6))
    two = filter_bank(ScienceImage(b, .4, mask=mask), geometry(), (.4, .8, 1.6))
    for name in one:
        np.testing.assert_allclose(one[name].score, two[name].score, equal_nan=True)
        assert np.array_equal(one[name].valid, two[name].valid)
    np.testing.assert_array_equal(a[mask], 1e12)  # inputs immutable


def test_science_grayscale_keeps_flux_units_and_display_is_separate():
    data = np.arange(160 * 160).reshape(160, 160).astype(float)
    original = data.copy()
    science = ScienceImage(data, .4, unit="nanomaggy")
    result = filter_bank(science, geometry())
    assert result["native_intensity"].metadata["input_unit"] == "nanomaggy"
    for method in ("linear", "asinh", "binary", "clahe"):
        out = display_transform(data, method)
        assert np.nanmin(out) >= 0 and np.nanmax(out) <= 1
    np.testing.assert_array_equal(data, original)
    rgb = np.stack((data, data * 0, data * 0), axis=-1)
    np.testing.assert_allclose(display_grayscale(rgb), .2126 * data)


def test_positive_ridge_exceeds_smooth_null_and_ring_is_not_called_arm():
    yy, xx = np.indices((160, 160)); r = np.hypot(xx - 79.5, yy - 79.5)
    smooth = 10 + np.exp(-r / 25)
    ridge = smooth + .5 * np.exp(-.5 * ((xx - 105) / 1.5) ** 2) * (abs(yy - 80) < 30)
    a = filter_bank(ScienceImage(smooth, .4), geometry())
    b = filter_bank(ScienceImage(ridge, .4), geometry())
    assert np.nanmax(b["bright_ridge"].score) > 5 * np.nanmax(a["bright_ridge"].score)
    assert all("not photometry" in result.metadata["interpretation"] for result in b.values())
    # No classifier or pitch gets created simply by producing a ridge score.
    assert all("pitch" not in result.metadata and "arm" not in result.metadata for result in b.values())


def test_background_rejects_masked_stars_and_nan():
    rng = np.random.default_rng(2)
    data = rng.normal(11, 2, (200, 200)); data[:10, :10] = np.nan
    data[-10:, -10:] = 1e5
    med, noise = robust_background(ScienceImage(data, .4))
    assert med == pytest.approx(11, abs=.1)
    assert noise == pytest.approx(2, abs=.1)


def test_ap_respects_common_masks_and_does_not_choose_test_threshold():
    truth = np.array([[0, 1, 0], [1, 0, 0]], bool)
    score = np.array([[0., 4, 1], [3, 0, 2]])
    valid = np.ones_like(truth)
    result = evaluate_score(score, truth, valid)
    assert result["average_precision"] == 1
    assert "threshold" not in result and "precision" not in result
    fixed = evaluate_score(score, truth, valid, threshold=2.5)
    assert fixed["precision"] == fixed["recall"] == 1
    assert evaluate_score(score, np.zeros_like(truth), valid)["average_precision"] is None
    assert evaluate_score(score, truth, np.zeros_like(truth))["valid_pixels"] == 0


def test_external_geometry_defines_identical_common_annulus():
    img = ScienceImage(np.random.default_rng(8).normal(0, 1, (160, 160)), .4)
    geo = geometry()
    maps = filter_bank(img, geo)
    expected = annulus_mask(img, geo)
    for item in maps.values():
        np.testing.assert_array_equal(item.valid, expected)
    with pytest.raises(ValueError, match="undersampled"):
        filter_bank(img, geo, (.01, .02, .03))
    with pytest.raises(ValueError, match="Inclination"):
        DiskGeometry((80, 80), 0, 85, 1, 10)


def test_compact_star_detection_protects_nucleus_and_accepts_manual_mask():
    yy, xx = np.indices((160, 160)); sigma = 1.8
    data = np.random.default_rng(4).normal(0, .01, (160, 160))
    for x, y in ((80, 80), (120, 45)):
        data += 2 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    manual = np.zeros_like(data, bool); manual[5:8, 5:8] = True
    mask, records = foreground_point_mask(ScienceImage(data, .4), (80, 80),
        psf_fwhm_arcsec=sigma * 2.354820045 * .4, manual_mask=manual)
    assert mask[45, 120] and not mask[80, 80] and mask[6, 6]
    assert any(abs(record["x_px"] - 120) <= 1 for record in records)


def test_psf_broadening_conserves_constant_and_propagates_variance():
    shape = (100, 100)
    source = ScienceImage(np.full(shape, 7.), .4, variance=np.full(shape, 4.),
                           psf_fwhm_arcsec=.8)
    target = match_gaussian_psf(source, 1.6)
    good = target.valid
    np.testing.assert_allclose(target.data[good], 7, atol=1e-10)
    assert np.median(target.variance[good]) < .2
    assert target.metadata["correlated_noise"]
    with pytest.raises(ValueError, match="sharpen"):
        match_gaussian_psf(source, .5)
    with pytest.raises(ValueError, match="input PSF"):
        match_gaussian_psf(ScienceImage(source.data, .4), 2.)


def test_reprojection_subpixel_shift_and_squared_variance():
    shape = (80, 80)
    yy, xx = np.indices(shape)
    data = xx + 2 * yy
    source = ScienceImage(data, .4, variance=np.full(shape, 4.), wcs=wcs_for(shape),
                          psf_fwhm_arcsec=1., unit="Jy/sr")
    target = reproject_to(source, wcs_for(shape, shift=(.5, .5)), shape, .4)
    # Each target pixel interpolates four neighbours with weight .25.
    np.testing.assert_allclose(target.data[5:-5, 5:-5], data[5:-5, 5:-5] - 1.5, atol=1e-6)
    np.testing.assert_allclose(target.variance[5:-5, 5:-5], 1., atol=1e-6)
    assert target.metadata["correlated_noise"]
    assert target.psf_fwhm_arcsec is None
    assert not target.metadata["photometry_validated"]


def test_registered_inverse_variance_coadd_uses_known_noise_and_shared_units():
    shape = (40, 40); wcs = wcs_for(shape)
    a = ScienceImage(np.full(shape, 4.), .4, variance=np.full(shape, 1.),
                     wcs=wcs, psf_fwhm_arcsec=1., unit="nanomaggy", band="g")
    b = ScienceImage(np.full(shape, 10.), .4, variance=np.full(shape, 4.),
                     wcs=wcs, psf_fwhm_arcsec=1., unit="nanomaggy", band="r")
    combo = inverse_variance_coadd([a, b])
    np.testing.assert_allclose(combo.data, 5.2)
    np.testing.assert_allclose(combo.variance, .8)
    assert combo.metadata["coadd_kind"] == "inverse variance detection image"
    b.wcs = wcs_for(shape, shift=(1, 0))
    with pytest.raises(ValueError, match="WCS grids differ"):
        inverse_variance_coadd([a, b])
    b.wcs = wcs; b.unit = "Jy"
    with pytest.raises(ValueError, match="matching calibrated"):
        inverse_variance_coadd([a, b])


def test_pixel_flux_reprojection_accounts_for_pixel_area():
    shape = (80, 80)
    source = ScienceImage(np.full(shape, 2.), .4, variance=np.ones(shape),
        wcs=wcs_for(shape), pixel_value_kind="flux_per_pixel")
    target = reproject_to(source, wcs_for((40, 40), scale=.8), (40, 40), .8)
    assert np.median(target.data[target.valid]) == pytest.approx(8.)


def test_robust_outer_isophote_refines_known_smooth_disk():
    shape = (180, 180); yy, xx = np.indices(shape); angle = np.radians(36)
    xp = (xx - 89.5) * np.cos(angle) + (yy - 89.5) * np.sin(angle)
    yp = -(xx - 89.5) * np.sin(angle) + (yy - 89.5) * np.cos(angle)
    data = .1 + 4 * np.exp(-np.hypot(xp, yp / .65) / 27)
    initial = DiskGeometry((89.5, 89.5), 30, np.degrees(np.arccos(.72)), 8, 28)
    refined, diagnostic = refine_outer_geometry(ScienceImage(data, .4), initial)
    assert diagnostic["success"]
    assert refined.pa_deg == pytest.approx(36, abs=1)
    assert np.cos(np.radians(refined.inclination_deg)) == pytest.approx(.65, abs=.03)


def test_bilinear_reprojection_discloses_compact_source_aliasing():
    # The point-sampled registration helper must never be marketed as a
    # photometric flux-conserving resampler. A narrow off-grid source exposes it.
    shape = (80, 80)
    data = np.zeros(shape); data[39, 39] = 1.
    source = ScienceImage(data, .4, wcs=wcs_for(shape),
        psf_fwhm_arcsec=.4, pixel_value_kind="flux_per_pixel")
    target = reproject_to(source, wcs_for((40, 40), scale=.8, shift=(.25, .25)), (40, 40), .8)
    assert abs(np.nansum(target.data) - 1.) > .25
    assert not target.metadata["photometry_validated"]
    assert target.psf_fwhm_arcsec is None


def test_coadd_rejects_shear_invisible_on_image_diagonal():
    shape = (40, 40)
    a = ScienceImage(np.ones(shape), .4, variance=np.ones(shape),
        wcs=wcs_for(shape), psf_fwhm_arcsec=1., unit="Jy/sr")
    sheared = wcs_for(shape)
    sheared.wcs.pc = [[1.1, -.1], [.1, .9]]
    b = ScienceImage(np.ones(shape), .4, variance=np.ones(shape),
        wcs=sheared, psf_fwhm_arcsec=1., unit="Jy/sr")
    with pytest.raises(ValueError, match="WCS grids differ"):
        inverse_variance_coadd([a, b])
