"""Injection and invariance tests of the actual measurement algorithms."""
import numpy as np
import pytest
from scipy.ndimage import zoom

from galaxy_lab.morphology import (estimate_background, inclination_from_q,
    load_fits, logpolar_fourier, measure_image, projected_pair_separation,
    sky_center_and_pa, synthetic_spiral)


def _kwargs(truth):
    return {k: v for k, v in truth.items() if k not in ("pitch_deg", "mode_m")}


@pytest.mark.parametrize("pitch,mode,inc,pa,psf,noise", [
    (12, 2, 0, 10, .8, .01), (22, 2, 35, 25, 1.2, .03),
    (32, 3, 55, 110, 1.5, .04), (42, 4, 30, 77, 1.5, .025),
])
def test_injected_pitch_and_arm_periodicity(pitch, mode, inc, pa, psf, noise):
    image, truth = synthetic_spiral(pitch_deg=pitch, mode_m=mode,
        inclination_deg=inc, pa_deg=pa, psf_sigma_px=psf, noise_sigma=noise)
    result = measure_image(image, sensitivity=False, **_kwargs(truth))
    assert result["spiral"]["mode_m"] == mode
    assert abs(result["spiral"]["pitch_abs_deg"] - pitch) < 1.2


def test_rotation_and_resampling_do_not_create_a_distance_measurement():
    image, truth = synthetic_spiral(noise_sigma=.005)
    original = measure_image(image, pixel_scale_arcsec=.4, sensitivity=False, **_kwargs(truth))
    rotated_truth = {**_kwargs(truth), "pa_deg": truth["pa_deg"] - 90}
    rotated = measure_image(np.rot90(image), sensitivity=False, **rotated_truth)
    factor = .65
    smaller = zoom(image, factor, order=1)
    scaled_truth = {**_kwargs(truth), "center_xy": ((smaller.shape[1] - 1) / 2, (smaller.shape[0] - 1) / 2),
                    "r_inner_px": truth["r_inner_px"] * factor,
                    "r_outer_px": truth["r_outer_px"] * factor}
    resampled = measure_image(smaller, pixel_scale_arcsec=.4 / factor,
                             sensitivity=False, **scaled_truth)
    assert rotated["spiral"]["pitch_abs_deg"] == pytest.approx(original["spiral"]["pitch_abs_deg"], abs=.2)
    assert resampled["spiral"]["pitch_abs_deg"] == pytest.approx(original["spiral"]["pitch_abs_deg"], abs=.5)
    assert resampled["r50_aperture_arcsec"] == pytest.approx(original["r50_aperture_arcsec"], rel=.025)
    assert original["distance_mpc_external"] is None
    assert "not identifiable" in original["distance_inference"]


def test_physical_scale_requires_external_distance_and_scales_linearly():
    image, truth = synthetic_spiral()
    a = measure_image(image, pixel_scale_arcsec=.4, distance_mpc=10,
                      sensitivity=False, **_kwargs(truth))
    b = measure_image(image, pixel_scale_arcsec=.4, distance_mpc=20,
                      sensitivity=False, **_kwargs(truth))
    assert b["r50_kpc_at_external_distance"] == 2 * a["r50_kpc_at_external_distance"]
    assert b["spiral"]["pitch_abs_deg"] == a["spiral"]["pitch_abs_deg"]


def test_masked_pixels_do_not_poison_analysis_but_large_gaps_fail():
    image, truth = synthetic_spiral()
    image[10:14, 10:14] = np.nan
    result = measure_image(image, sensitivity=False, **_kwargs(truth))
    assert np.isfinite(result["background"])
    image[:, :128] = np.nan
    with pytest.raises(ValueError, match="coverage"):
        logpolar_fourier(image, truth["center_xy"], truth["pa_deg"],
                         truth["inclination_deg"], truth["r_inner_px"], truth["r_outer_px"])


def test_sensitivity_is_labeled_and_rejects_edge_on():
    image, truth = synthetic_spiral()
    result = measure_image(image, **_kwargs(truth))
    assert len(result["sensitivity_trials"]) == 6
    assert any("not a confidence" in line for line in result["limitations"])
    with pytest.raises(ValueError, match="inclination"):
        measure_image(image, inclination_deg=85)


def test_thickness_inclination_and_background_robustness():
    assert inclination_from_q(.5) == pytest.approx(60.)
    assert inclination_from_q(1., .2) == 0
    rng = np.random.default_rng(22)
    sky = rng.normal(17, 2, (256, 256))
    sky[0, :30] = 10000
    sky[0, 100:120] = np.nan
    background, noise = estimate_background(sky)
    assert background == pytest.approx(17, abs=.06)
    assert noise == pytest.approx(2, abs=.08)


def test_fits_wcs_native_pixels_and_sky_conventions(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [33, 33]
    w.wcs.cdelt = [-.4 / 3600, .4 / 3600]
    w.wcs.crval = [202.4, 47.2]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = np.arange(64 * 64).reshape(64, 64).astype(float)
    data[0, 0] = np.nan
    path = tmp_path / "native.fits"
    fits.writeto(path, data, w.to_header())
    image = load_fits(path)
    assert np.array_equal(image.data, data, equal_nan=True)
    assert image.metadata["pixel_scale_arcsec"] == pytest.approx(.4)
    center, pa = sky_center_and_pa(image.metadata, 202.4, 47.2, 0)
    assert center == pytest.approx((32, 32), abs=1e-7)
    assert pa == pytest.approx(90, abs=1e-4)


def test_sky_pair_separation_requires_distance_for_physical_units():
    angular = projected_pair_separation(0, 0, 1 / 3600, 0)
    assert angular["angular_separation_arcsec"] == pytest.approx(1.)
    assert "projected_separation_kpc_small_angle" not in angular
    physical = projected_pair_separation(0, 0, 1 / 3600, 0, distance_mpc=10)
    assert physical["projected_separation_kpc_small_angle"] == pytest.approx(.04848136811)


def test_sample_selection_does_not_read_redshift_or_photometry(tmp_path):
    import csv
    from experiments.sample_image_features import select_catalog
    path = tmp_path / "sample.csv"
    fields = ["galaxy_id", "specObjID", "ra", "dec", "angular_r50_arcsec", "spec_z", "mag_r"]
    rows = [{"galaxy_id": str(10 ** 18 + k), "specObjID": str(2 * 10 ** 18 + k),
             "ra": 20 + k, "dec": 10, "angular_r50_arcsec": 4, "spec_z": .02, "mag_r": 17}
            for k in range(3)]
    def write():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write()
    original = select_catalog(path, count=2)
    for row in rows:
        row["spec_z"], row["mag_r"] = "not used", "not used"
    write()
    assert select_catalog(path, count=2) == original
    assert isinstance(original[0]["galaxy_id"], str)
    assert "spec_z" not in original[0]


def test_sample_null_image_rejects_all_arm_features(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    from experiments.sample_image_features import ARM_KEYS, extract_one
    yy, xx = np.indices((256, 256))
    image = .2 + np.exp(-np.hypot(xx - 127.5, yy - 127.5) / 45)
    w = WCS(naxis=2)
    w.wcs.crpix = [128.5, 128.5]
    w.wcs.cdelt = [-.396 / 3600, .396 / 3600]
    w.wcs.crval = [200, 20]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    path = tmp_path / "null.fits"
    fits.writeto(path, image, w.to_header())
    row, _ = extract_one({"selection_row": 0, "galaxy_id": "1000000000000000001",
        "specObjID": "2000000000000000001", "ra_deg": 200, "dec_deg": 20,
        "angular_r50_arcsec": 10, "acquisition_status": "cached", "path": str(path)})
    assert row["extraction_status"] == "candidate_rejected"
    assert row["image_available"] == 1
    assert all(row[key] is None for key in ARM_KEYS)
