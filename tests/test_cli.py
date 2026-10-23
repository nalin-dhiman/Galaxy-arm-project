"""Check the physical meaning of the public command interface."""
import json

import pytest

from galaxy_lab.__main__ import main


def test_pair_keeps_angular_measurement_independent_of_external_distance(capsys):
    angular_args = ["pair", "--ra1", "0", "--dec1", "0", "--ra2", str(1/3600), "--dec2", "0"]
    main(angular_args)
    angular = json.loads(capsys.readouterr().out)
    assert angular["angular_separation_arcsec"] == pytest.approx(1)
    assert not any("kpc" in key for key in angular)
    main(angular_args + ["--distance-mpc", "1"])
    physical = json.loads(capsys.readouterr().out)
    assert physical["angular_separation_arcsec"] == pytest.approx(1)
    assert physical["projected_separation_kpc_small_angle"] == pytest.approx(0.004848136811)


def test_cli_fluid_origin_is_regular_and_supplies_no_viscosity_measurement(capsys):
    main(["fluid", "--rho0", "10000000", "--sigma", "100", "--radii", "0", "10"])
    result = json.loads(capsys.readouterr().out)
    assert result["density_msun_kpc3"][0] == 1e7
    assert result["enclosed_mass_msun"][0] == result["circular_speed_kms"][0] == 0
    assert "no viscosity is inferred" in result["interpretation"]


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_cli_pair_rejects_nonfinite_physical_distance(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        main(["pair", "--ra1", "0", "--dec1", "0", "--ra2", "1", "--dec2", "0",
              "--distance-mpc=" + bad])


def _write_filter_input(tmp_path, *, unit="nanomaggy", with_wcs=True, shape=(112, 160)):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    yy, xx = np.indices(shape)
    data = .13 + np.exp(-np.hypot((xx - 79.5), (yy - 55.5) / .8) / 27)
    data += np.random.default_rng(8).normal(0, .015, shape)
    header = fits.Header()
    if with_wcs:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
        wcs.wcs.cdelt = [-.4 / 3600, .4 / 3600]
        wcs.wcs.crval = [150, 20]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        header.update(wcs.to_header())
    if unit is not None:
        header["BUNIT"] = unit
    header["FILTER"] = "r"
    path = tmp_path / "rectangular.fits"
    invvar = np.full(shape, 1 / .015 ** 2)
    invvar[0, 0] = 0
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header, name="SCI"),
                  fits.ImageHDU(invvar, name="IVAR")]).writeto(path)
    return path, data


def test_filter_cli_exports_nonsquare_wcs_scores_and_separate_display(tmp_path, capsys):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image
    source, data = _write_filter_input(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "filtered"
    main(["filter", str(source), "--hdu", "SCI", "--invvar-hdu", "IVAR",
          "--output-dir", str(output), "--outer-arcsec", "17", "--inner-arcsec", "4",
          "--inclination-deg", "35", "--pa-deg", "20"])
    result = json.loads(capsys.readouterr().out)
    assert result["input_shape"] == [112, 160]
    assert result["input_unit"] == "nanomaggy"
    assert result["input_calibration_verified"] is False
    assert result["pixel_scale_arcsec"] == pytest.approx(.4)
    assert result["geometry"]["center_xy"] == [79.5, 55.5]
    assert source.read_bytes() == original
    assert len(list(output.glob("*.fits"))) == 11
    valid = fits.getdata(output / "common_valid_mask.fits").astype(bool)
    assert valid.shape == data.shape and valid.sum() >= 100
    for name in result["filters"]:
        scores = fits.getdata(output / f"{name}.fits")
        assert np.isfinite(scores[valid]).all()
        assert np.isnan(scores[~valid]).all()
        header = fits.getheader(output / f"{name}.fits")
        assert header["BUNIT"] == "structure score"
        np.testing.assert_allclose(WCS(header).celestial.pixel_to_world_values(79.5, 55.5), [150, 20], atol=1e-8)
    for name in ("grayscale", "asinh", "clahe", "binary"):
        assert Image.open(output / f"{name}.png").size == (160, 112)
    assert (output / "filter_gallery.png").exists()
    assert json.loads((output / "metadata.json").read_text())["input_unit"] == "nanomaggy"


def test_filter_cli_missing_bunit_is_explicit_and_override_scale_works(tmp_path, capsys):
    source, _ = _write_filter_input(tmp_path, unit=None, with_wcs=False)
    output = tmp_path / "unknown_units"
    main(["filter", str(source), "--hdu", "1", "--pixel-scale-arcsec", ".4",
          "--band", "unknown native channel", "--output-dir", str(output),
          "--outer-arcsec", "17"])
    result = json.loads(capsys.readouterr().out)
    assert result["input_unit"] == "unspecified"
    assert "unknown" in result["unit_status"]
    assert result["pixel_scale_source"] == "explicit override"
    assert result["input_calibration_verified"] is False


@pytest.mark.parametrize("extra", [
    ["--pixel-scale-arcsec", "0"],
    ["--pixel-scale-arcsec", "nan"],
    ["--center-x", "30"],
    ["--center-x", "500", "--center-y", "40"],
    ["--inner-arcsec", "20", "--outer-arcsec", "10"],
    ["--inclination-deg", "85"],
    ["--scales-arcsec", ".8", ".4", "1.6"],
    ["--scales-arcsec", ".01", ".02", ".03"],
    ["--hdu", "99"],
])
def test_filter_cli_rejects_invalid_entries_before_writing(tmp_path, extra):
    source, _ = _write_filter_input(tmp_path)
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        main(["filter", str(source), "--hdu", "1", "--output-dir", str(output)] + extra)
    assert error.value.code == 2
    assert not output.exists()


def test_filter_cli_preserves_existing_output_and_requires_native_2d(tmp_path):
    from astropy.io import fits
    import numpy as np
    source, _ = _write_filter_input(tmp_path)
    output = tmp_path / "keep"; output.mkdir()
    sentinel = output / "keep.txt"; sentinel.write_text("original")
    with pytest.raises(SystemExit):
        main(["filter", str(source), "--hdu", "1", "--output-dir", str(output)])
    assert sentinel.read_text() == "original"
    cube = tmp_path / "cube.fits"
    fits.writeto(cube, np.zeros((3, 100, 160)))
    with pytest.raises(SystemExit):
        main(["filter", str(cube), "--output-dir", str(tmp_path / "cube_out")])


def test_filter_cli_requires_scale_when_wcs_is_absent(tmp_path):
    source, _ = _write_filter_input(tmp_path, with_wcs=False)
    with pytest.raises(SystemExit):
        main(["filter", str(source), "--hdu", "1", "--output-dir", str(tmp_path / "no_wcs")])
