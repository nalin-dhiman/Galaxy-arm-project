import hashlib
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from galaxy_lab.native import read_native_grz


def make_native(tmp_path, shift_invvar=False):
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [150., 20.]; wcs.wcs.crpix = [40., 32.]
    wcs.wcs.cdelt = [-.262/3600, .262/3600]; wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdus, psfs = [fits.PrimaryHDU()], []
    yy, xx = np.indices((21, 21)); stamp = np.exp(-((xx-10)**2+(yy-10)**2)/8)
    stamp /= stamp.sum()
    for band in "grz":
        header = wcs.to_header(); header["BAND"] = band; header["IMAGETYP"] = "image"
        hdus.append(fits.ImageHDU(np.ones((64, 80)), header))
        header = header.copy(); header["IMAGETYP"] = "invvar"
        if shift_invvar:
            header["CRPIX1"] += 5.
        hdus.append(fits.ImageHDU(np.full((64, 80), 4.), header))
        ph = fits.Header({"BAND": band})
        psfs.append(fits.PrimaryHDU(stamp, ph) if not psfs else fits.ImageHDU(stamp, ph))
    image, psf = tmp_path / "native.fits", tmp_path / "psf.fits"
    fits.HDUList(hdus).writeto(image); fits.HDUList(psfs).writeto(psf)
    return image, psf


def test_native_read_preserves_flux_and_reports_actual_psf_and_noise(tmp_path):
    image, psf = make_native(tmp_path)
    before = hashlib.sha256(image.read_bytes()).hexdigest()
    images, metadata = read_native_grz(image, psf, declared_unit="nanomaggy/pixel")
    assert np.array_equal(images["r"].data, np.ones((64, 80)))
    assert images["r"].variance[10, 10] == pytest.approx(.25)
    assert images["r"].psf_fwhm_arcsec == pytest.approx(2.354820045 * 2 * .262, rel=2e-5)
    assert metadata["quality_features"]["apparent_noise_g_nanomaggy"] == pytest.approx(.5)
    assert hashlib.sha256(image.read_bytes()).hexdigest() == before


def test_wrong_inverse_variance_grid_is_rejected_even_if_shape_matches(tmp_path):
    image, psf = make_native(tmp_path, True)
    with pytest.raises(ValueError, match="Inverse-variance WCS"):
        read_native_grz(image, psf, declared_unit="nanomaggy/pixel")


def test_rendered_or_undeclared_units_cannot_become_science_flux(tmp_path):
    image, psf = make_native(tmp_path)
    with pytest.raises(ValueError, match="calibration"):
        read_native_grz(image, psf, declared_unit="RGB counts")


def test_explicit_declaration_cannot_override_contradictory_header_units(tmp_path):
    image, psf = make_native(tmp_path)
    with fits.open(image, mode="update") as h:
        h[1].header["BUNIT"] = "erg/s/cm2"
    with pytest.raises(ValueError, match="BUNIT contradicts"):
        read_native_grz(image, psf, declared_unit="nanomaggy/pixel")


def test_mask_wcs_checked_when_supplied(tmp_path):
    image, psf = make_native(tmp_path)
    header = fits.getheader(image, 1)
    header["CRPIX2"] += 3.
    mask = tmp_path / "mask.fits"
    fits.writeto(mask, np.zeros((64, 80), np.uint8), header)
    with pytest.raises(ValueError, match="mask WCS"):
        read_native_grz(image, psf, declared_unit="nanomaggy/pixel", mask_path=mask)


def test_neighboring_last_hdus_cannot_replace_the_target_tile(tmp_path):
    image, psf = make_native(tmp_path)
    with fits.open(image, mode="append") as h:
        for band in "grz":
            header = h[1].header.copy(); header["BAND"] = band
            header["CRVAL1"] += 1.; header["IMAGETYP"] = "image"
            h.append(fits.ImageHDU(np.full((64, 8), 999.), header))
            header = header.copy(); header["IMAGETYP"] = "invvar"
            h.append(fits.ImageHDU(np.ones((64, 8)), header))
    target = WCS(fits.getheader(image, 1)).celestial.pixel_to_world(30., 20.)
    with pytest.raises(ValueError, match="Multiple native tiles"):
        read_native_grz(image, psf, declared_unit="nanomaggy/pixel")
    images, meta = read_native_grz(image, psf, declared_unit="nanomaggy/pixel", target_sky=target)
    assert images["r"].data.shape == (64, 80)
    assert images["r"].data.max() == 1.
    assert meta["selected_hdus"]["r"]["image_hdu"] == 3
    assert meta["native_grids_in_file"] == 2


def test_target_coordinate_selects_a_smaller_correct_tile_over_large_neighbor(tmp_path):
    image, psf = make_native(tmp_path)
    with fits.open(image, mode="append") as h:
        header = h[1].header.copy(); header["CRVAL1"] += 1.
        target = WCS(header).celestial.pixel_to_world(30., 20.)
        for band in "grz":
            header["BAND"] = band; header["IMAGETYP"] = "image"
            h.append(fits.ImageHDU(np.full((64, 40), 7.), header.copy()))
            header["IMAGETYP"] = "invvar"
            h.append(fits.ImageHDU(np.ones((64, 40)), header.copy()))
    images, meta = read_native_grz(image, psf, declared_unit="nanomaggy/pixel", target_sky=target)
    assert images["r"].data.shape == (64, 40)
    assert images["r"].data.max() == 7.


def test_identical_duplicate_planes_resolve_consistently_and_conflicts_fail(tmp_path):
    image, psf = make_native(tmp_path)
    with fits.open(image, mode="append") as h:
        for index in range(1, 7):
            h.append(fits.ImageHDU(h[index].data.copy(), h[index].header.copy()))
    images, meta = read_native_grz(image, psf, declared_unit="nanomaggy/pixel")
    assert images["r"].data.max() == 1.
    assert meta["selected_hdus"]["r"]["image_hdu"] == 9
    with fits.open(image, mode="update") as h:
        h[9].data[0, 0] = 55.
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        read_native_grz(image, psf, declared_unit="nanomaggy/pixel")
