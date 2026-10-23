"""Image-only inference, schema checks, source masking and model persistence."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from galaxy_lab.filtering import ScienceImage, DiskGeometry
from galaxy_lab.segmentation import (arm_features, predict_arm_regions,
    probability_descriptors, predict_feature_probabilities)


def case(seed=5):
    shape = (160, 160); yy, xx = np.indices(shape)
    radius = np.hypot(xx - 79.5, yy - 79.5)
    theta = np.arctan2(yy - 79.5, xx - 79.5)
    arms = .2 * np.exp(-radius / 40) * np.cos(2 * theta - 5 * np.log(np.maximum(radius, 1)))
    images = {}
    for index, band in enumerate('grz'):
        data = .1 + np.exp(-radius / 35) + arms * (1 - .15 * index)
        data += np.random.default_rng(seed + index).normal(0, .02, shape)
        images[band] = ScienceImage(data, .4, band=band, unit="nanomaggy/pixel")
    geometry = DiskGeometry((79.5, 79.5), 0, 0, 4, 23, "independently supplied test geometry")
    return images, geometry


def test_feature_api_is_label_free_and_masked_values_never_contribute():
    images, geometry = case()
    mask = np.zeros((160, 160), bool); mask[70:73, 110:113] = True
    for im in images.values():
        im.mask = mask.copy(); im.data[mask] = 1e20
    one = arm_features(images, geometry)
    for im in images.values():
        im.data[mask] = np.nan
    two = arm_features(images, geometry)
    np.testing.assert_allclose(one.X, two.X)
    np.testing.assert_array_equal(one.valid, two.valid)
    assert len(one.feature_names) == 44
    assert not any("distance" in name or "mass" in name or "redshift" in name or "truth" in name for name in one.feature_names)


def test_single_image_features_work_but_multiband_model_requires_its_bands(tmp_path):
    images, geometry = case()
    full = arm_features(images, geometry)
    single = arm_features(images['g'], geometry)
    assert single.metadata['bands'] == ('g',)
    estimator = LogisticRegression().fit(full.X[:, :3], full.X[:, 0] > .6)
    artifact = dict(feature_names=full.feature_names, feature_indices=[0, 1, 2],
        estimator=estimator, calibrator=None, threshold=.5,
        scales_sigma_arcsec=[.4, .8, 1.6], selected_model="unit-test-radius")
    with pytest.raises(ValueError, match="schema/bands"):
        predict_feature_probabilities(artifact, single)
    import joblib
    path = tmp_path / 'model.joblib'; joblib.dump(artifact, path)
    original = {b: im.data.copy() for b, im in images.items()}
    result = predict_arm_regions(path, images, geometry)
    assert np.all((result['probability'][result['valid']] >= 0) & (result['probability'][result['valid']] <= 1))
    assert np.isnan(result['probability'][~result['valid']]).all()
    assert not result['binary'][~result['valid']].any()
    for band in images:
        np.testing.assert_array_equal(original[band], images[band].data)


def test_circular_feature_map_is_rotation_equivariant():
    images, geometry = case()
    original = arm_features(images, geometry)
    rotated = {b: ScienceImage(np.rot90(im.data), .4, band=b, unit=im.unit) for b, im in images.items()}
    other = arm_features(rotated, geometry)
    for index, name in enumerate(original.feature_names):
        field = np.full(original.valid.shape, np.nan); field[original.valid] = original.X[:, index]
        other_field = np.full(other.valid.shape, np.nan); other_field[other.valid] = other.X[:, index]
        np.testing.assert_allclose(np.rot90(field), other_field, atol=2e-5, rtol=2e-5, equal_nan=True, err_msg=name)


def test_feature_restriction_and_descriptors_require_no_annotation():
    images, geometry = case()
    features = arm_features(images, geometry)
    keep = np.indices(features.valid.shape)[1] < 80
    restricted = features.restrict(keep)
    assert restricted.X.shape[0] == (features.valid & keep).sum()
    probability = np.linspace(0, 1, len(features.X))
    desc = probability_descriptors(probability, features, .5)
    assert desc['arm_probability_mean'] == pytest.approx(.5)
    assert desc['arm_candidate_area_fraction'] == pytest.approx(.5, abs=1e-3)
    with pytest.raises(ValueError, match="Probability"):
        probability_descriptors(np.full(len(features.X), 1.1), features, .5)


def test_rejects_mismatched_grid_display_and_unknown_crossband_units():
    images, geometry = case()
    images['g'].pixel_scale_arcsec = .5
    with pytest.raises(ValueError, match="same registered"):
        arm_features(images, geometry)
    images['g'].pixel_scale_arcsec = .4
    images['g'].provenance = 'rendered_display'
    with pytest.raises(ValueError, match="native science"):
        arm_features(images, geometry)
    images['g'].provenance = 'science'; images['g'].unit = 'unspecified'
    with pytest.raises(ValueError, match="photometric units"):
        arm_features(images, geometry)


def test_balanced_sampling_weights_each_galaxy_equally():
    from experiments.supervised_arm_experiment import training_samples
    from galaxy_lab.segmentation import ArmFeatureSet
    objects = []
    for index, positive in enumerate((7, 250, 900)):
        X = np.arange(2000, dtype=np.float32)[:,None]
        y = np.arange(2000) < positive
        features = ArmFeatureSet(X, np.ones((40,50),bool), ('fake_image_feature',), {}, {})
        objects.append({'annotation': {'id': str(index)}, 'features': features, 'y3': y})
    X, y, weights, owners = training_samples(objects)
    for index in range(3):
        chosen = owners == str(index)
        assert weights[chosen].sum() == pytest.approx(800)
        assert y[chosen].mean() == .5
    assert X.shape[0] == len(weights) == len(owners)


def test_equal_sky_group_interval_has_a_different_estimand_from_galaxy_mean():
    from experiments.supervised_arm_experiment import sky_group_interval
    rows = [dict(sky_group="A", model_AP=.2, raw_g_AP=.1) for _ in range(9)]
    rows += [dict(sky_group="B", model_AP=1., raw_g_AP=.1)]
    result = sky_group_interval(rows, "raw_g_AP")
    assert result["mean_difference"] == pytest.approx(.5)
    assert result["mean_difference"] != pytest.approx(np.mean([r["model_AP"]-r["raw_g_AP"] for r in rows]))
    assert result["resampling_unit"] == "sky_group"


def test_manifest_reader_uses_explicit_target_hdus_before_neighbor_strips(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    from experiments.supervised_arm_experiment import read_manifest_native
    wcs=WCS(naxis=2);wcs.wcs.crpix=[64,48];wcs.wcs.cdelt=[-.4/3600,.4/3600]
    wcs.wcs.crval=[150,20];wcs.wcs.ctype=["RA---TAN","DEC--TAN"]
    hdus=[fits.PrimaryHDU()]
    for brick,shape in (("TARGET",(96,128)),("NEIGHBOR",(96,20))):
        for band in 'grz':
            for kind in ('image','invvar'):
                header=wcs.to_header();header['BRICK']=brick;header['BAND']=band;header['IMAGETYP']=kind
                hdus.append(fits.ImageHDU(np.ones(shape),header))
    path=tmp_path/'multi_brick.fits';fits.HDUList(hdus).writeto(path)
    records={b:dict(image_hdu=1+2*i,invvar_hdu=2+2*i,psf_fwhm_arcsec=1.) for i,b in enumerate('grz')}
    native=dict(path=str(path),bands=records,ra_deg=150.,dec_deg=20.)
    images=read_manifest_native(native)
    assert all(image.data.shape==(96,128) for image in images.values())
    assert all(image.metadata['brick']=='TARGET' for image in images.values())
    native['bands']={b:dict(image_hdu=7+2*i,invvar_hdu=8+2*i,psf_fwhm_arcsec=1.) for i,b in enumerate('grz')}
    with pytest.raises(ValueError,match='does not contain'):
        read_manifest_native(native)
