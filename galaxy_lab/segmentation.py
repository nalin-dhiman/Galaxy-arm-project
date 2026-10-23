"""Label-free angular image features and auditable supervised arm-region inference.

Outputs estimate agreement with a specified human annotation protocol, not
physical arm membership, a distance, or fluid microphysics. Native image values
are never overwritten. No label, redshift, mass or distance is a feature input.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from .filtering import (ScienceImage, DiskGeometry, radius_map, annulus_mask,
    radial_residual, robust_background, _nearest_fill)

FEATURE_VERSION = "arm-local-v1"


@dataclass
class ArmFeatureSet:
    X: np.ndarray
    valid: np.ndarray
    feature_names: tuple[str, ...]
    descriptors: dict
    metadata: dict

    def restrict(self, allowed):
        if np.shape(allowed) != self.valid.shape:
            raise ValueError("Restriction mask shape differs")
        retained = np.asarray(allowed, bool)[self.valid]
        valid = self.valid & np.asarray(allowed, bool)
        return ArmFeatureSet(self.X[retained], valid, self.feature_names,
                             dict(self.descriptors), {**self.metadata, "restricted_pixels": int(valid.sum())})


def _scale(values):
    med = float(np.median(values))
    scale = max(float(1.4826 * np.median(abs(values - med))),
                float(np.std(values)) * .1, 1e-12)
    return med, scale


def arm_features(images: dict[str, ScienceImage] | ScienceImage,
                 geometry: DiskGeometry, scales_arcsec=(.4, .8, 1.6)) -> ArmFeatureSet:
    """Fixed multiscale intensity/color/contrast features, independent of labels.

    A single image is supported for descriptive feature extraction. A fitted
    three-band model requires all its original bands; missing bands are never
    synthesized. Supplied geometry and manual foreground masks are conditional
    external inputs. All bands must already share a WCS/pixel grid; the function
    performs no science reprojection and no unmeasured PSF matching.
    """
    if isinstance(images, ScienceImage):
        images = {images.band if images.band != "unknown" else "single": images}
    if not images:
        raise ValueError("Supply at least one science image")
    bands = tuple(sorted(images))
    reference = images[bands[0]]
    scales = np.asarray(scales_arcsec, float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or not np.all(scales > 0) or not np.all(np.diff(scales) > 0):
        raise ValueError("Three finite positive increasing angular scales are required")
    sigma = scales / reference.pixel_scale_arcsec
    if sigma[0] < .6:
        raise ValueError("Finest scale is undersampled")
    common = annulus_mask(reference, geometry)
    for band, im in images.items():
        if im.data.shape != reference.data.shape or not np.isclose(im.pixel_scale_arcsec, reference.pixel_scale_arcsec):
            raise ValueError("Bands require the same registered native pixel grid")
        if im.provenance == "rendered_display":
            raise ValueError("This supervised feature schema requires native science data")
        common &= im.valid
        if im.wcs is not None and reference.wcs is not None:
            xx, yy = np.meshgrid(np.linspace(0, im.data.shape[1] - 1, 3),
                                 np.linspace(0, im.data.shape[0] - 1, 3))
            offset = im.wcs.pixel_to_world(xx, yy).separation(reference.wcs.pixel_to_world(xx, yy)).arcsec
            if not np.isfinite(offset).all() or np.max(offset) > .01 * reference.pixel_scale_arcsec:
                raise ValueError("Band WCS solutions differ")
    # One common guard across every feature/band, including holes and borders.
    from scipy.ndimage import distance_transform_edt
    inputs_valid = np.logical_and.reduce([im.valid for im in images.values()])
    safe = inputs_valid.copy()
    if not safe.all():
        safe &= distance_transform_edt(safe) > 4 * sigma[-1]
    border = int(np.ceil(4 * sigma[-1]))
    safe[:border] = safe[-border:] = False
    safe[:, :border] = safe[:, -border:] = False
    common &= safe
    if common.sum() < 100:
        raise ValueError("Fewer than 100 common feature pixels after fixed guards")
    radius = radius_map(reference.data.shape, geometry, reference.pixel_scale_arcsec)
    normalized_radius = radius / geometry.r_outer_arcsec
    yy, xx = np.indices(reference.data.shape)
    theta_image = np.arctan2(yy - geometry.center_xy[1], xx - geometry.center_xy[0])
    radial_cos2, radial_sin2 = np.cos(2 * theta_image), np.sin(2 * theta_image)
    maps = [normalized_radius, normalized_radius ** 2, np.log1p(normalized_radius)]
    names = ["radius_fraction", "radius_fraction_squared", "log_radius_fraction"]
    descriptors = {"analysis_valid_pixels": int(common.sum()),
                   "analysis_area_arcsec2": float(common.sum() * reference.pixel_scale_arcsec ** 2),
                   "external_annulus_inner_arcsec": geometry.r_inner_arcsec,
                   "external_annulus_outer_arcsec": geometry.r_outer_arcsec}
    signed_signals = {}
    for band in bands:
        im = images[band]
        bg, noise = robust_background(im)
        native = _nearest_fill(im.data - bg, im.valid)
        med, intensity_scale = _scale(native[common])
        residual, profile = radial_residual(im, geometry)
        residual = _nearest_fill(residual, im.valid)
        _, contrast_scale = _scale(residual[common])
        fine, middle, coarse = [gaussian_filter(residual, s) for s in sigma]
        hxx = gaussian_filter(residual, sigma[1], order=(0, 2)) * sigma[1] ** 2
        hyy = gaussian_filter(residual, sigma[1], order=(2, 0)) * sigma[1] ** 2
        hxy = gaussian_filter(residual, sigma[1], order=(1, 1)) * sigma[1] ** 2
        delta = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        l1, l2 = (hxx + hyy - delta) / 2, (hxx + hyy + delta) / 2
        small = np.where(abs(l1) <= abs(l2), l1, l2)
        large = np.where(abs(l1) <= abs(l2), l2, l1)
        ridge = np.maximum(-large, 0) * np.exp(-.5 * (small / np.maximum(abs(large), 1e-12) / .5) ** 2)
        # Bright-ridge tangent relative to the projected center-to-pixel axis.
        # Double angles remove the arbitrary sign of an eigenvector. These are
        # local image orientation cues, not a global spiral-pitch measurement.
        tangent_cos2 = (hxx - hyy) / np.maximum(delta, 1e-12)
        tangent_sin2 = 2 * hxy / np.maximum(delta, 1e-12)
        relative_cos2 = tangent_cos2 * radial_cos2 + tangent_sin2 * radial_sin2
        relative_sin2 = tangent_sin2 * radial_cos2 - tangent_cos2 * radial_sin2
        gx = gaussian_filter(residual, sigma[0], order=(0, 1)) * sigma[0]
        gy = gaussian_filter(residual, sigma[0], order=(1, 0)) * sigma[0]
        signal_floor = max(noise, intensity_scale * .1, 1e-12)
        local = [np.arcsinh(native / signal_floor), (native - med) / intensity_scale,
            residual / contrast_scale, (native - residual) / intensity_scale,
            (fine - middle) / contrast_scale, (middle - coarse) / contrast_scale,
            -(hxx + hyy) / contrast_scale, ridge / contrast_scale,
            np.hypot(gx, gy) / contrast_scale, fine / contrast_scale,
            coarse / contrast_scale, ridge / contrast_scale * relative_cos2,
            ridge / contrast_scale * relative_sin2]
        labels = ("asinh_snr", "brightness_z", "radial_contrast_z", "radial_profile_z",
                  "dog_fine_z", "dog_coarse_z", "negative_log_z", "ridge_z",
                  "gradient_z", "fine_contrast_z", "coarse_contrast_z",
                  "ridge_radial_cos2_weighted", "ridge_radial_sin2_weighted")
        maps.extend(local); names.extend(f"{band}_{label}" for label in labels)
        positive = np.maximum(native[common], 0)
        total = float(positive.sum())
        radial_order = np.argsort(radius[common])
        cumulative = np.cumsum(positive[radial_order])
        r50 = float(radius[common][radial_order][np.searchsorted(cumulative, .5 * total)]) if total > 0 else np.nan
        descriptors.update({f"{band}_annular_positive_flux": total,
            f"{band}_annular_signed_flux": float(native[common].sum()),
            f"{band}_annular_mean": float(np.mean(native[common])),
            f"{band}_annular_r50_arcsec": r50, f"{band}_background_noise": noise,
            f"{band}_brightness_p90": float(np.percentile(native[common], 90)),
            f"{band}_relative_contrast_rms": float(np.std(residual[common]) / max(np.mean(positive), 1e-12))})
        signed_signals[band] = (native, max(noise, 1e-12))
    if all(b in images for b in ("g", "r", "z")):
        units = {images[b].unit for b in ("g", "r", "z")}
        if len(units) != 1 or "unspecified" in units:
            raise ValueError("Cross-band features require matching declared photometric units")
        for left, right in (("g", "r"), ("r", "z")):
            a, an = signed_signals[left]; b, bn = signed_signals[right]
            floor = max(an, bn)
            # Signed asinh difference retains noisy/negative flux pixels.
            color = np.arcsinh(a / floor) - np.arcsinh(b / floor)
            maps.append(color); names.append(f"{left}{right}_asinh_color")
            descriptors[f"{left}{right}_annular_asinh_color"] = float(np.mean(color[common]))
    X = np.column_stack([values[common] for values in maps]).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite derived features remain")
    return ArmFeatureSet(X, common, tuple(names), descriptors, {
        "feature_version": FEATURE_VERSION, "bands": bands, "scales_sigma_arcsec": scales.tolist(),
        "geometry_source": geometry.source, "fixed_mask_guard_sigma": 4.,
        "interpretation": "image-only predictive features, no physical distance or arm truth",
        "input_units": {b: im.unit for b, im in images.items()}})


def predict_feature_probabilities(artifact: dict, features: ArmFeatureSet):
    if tuple(artifact["feature_names"]) != features.feature_names:
        raise ValueError("Image feature schema/bands differ from the fitted model")
    indices = artifact.get("feature_indices", np.arange(features.X.shape[1]))
    raw = artifact["estimator"].predict_proba(features.X[:, indices])[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        probability = calibrator.predict_proba(logits[:, None])[:, 1]
    else:
        probability = raw
    return np.clip(probability, 0, 1), np.clip(raw, 0, 1)


def probability_descriptors(probability, features: ArmFeatureSet, threshold: float):
    probability = np.asarray(probability, float)
    if probability.shape != (len(features.X),) or not np.isfinite(probability).all() or not np.all((probability >= 0) & (probability <= 1)):
        raise ValueError("Probability vector must match valid feature pixels and lie in [0,1]")
    radius_index = features.feature_names.index("radius_fraction")
    radius = features.X[:, radius_index]
    weight = probability.sum()
    return {"arm_probability_mean": float(np.mean(probability)),
        "arm_probability_p90": float(np.percentile(probability, 90)),
        "arm_probability_std": float(np.std(probability)),
        "arm_probability_entropy_mean": float(np.mean(-(np.clip(probability, 1e-12, 1) * np.log2(np.clip(probability, 1e-12, 1)) + np.clip(1 - probability, 1e-12, 1) * np.log2(np.clip(1 - probability, 1e-12, 1))))),
        "arm_candidate_area_fraction": float(np.mean(probability >= threshold)),
        "arm_probability_weighted_radius_fraction": float(np.dot(radius, probability) / weight) if weight > 0 else 0.,
        "arm_probability_inner_mean": float(np.mean(probability[radius < .55])) if (radius < .55).any() else 0.,
        "arm_probability_outer_mean": float(np.mean(probability[radius >= .55])) if (radius >= .55).any() else 0.}


def predict_arm_regions(model_path: str | Path | dict, images, geometry: DiskGeometry):
    """Infer without labels from a trusted locally produced model artifact."""
    if isinstance(model_path, dict):
        artifact = model_path
    else:
        import joblib
        artifact = joblib.load(model_path)
    if artifact.get("required_bands") and len(artifact["required_bands"]) > 1:
        if not isinstance(images, dict) or set(images) != set(artifact["required_bands"]):
            raise ValueError("The fitted model requires its full original band set")
        psfs = [im.psf_fwhm_arcsec for im in images.values()]
        if any(value is None or not np.isfinite(value) for value in psfs):
            raise ValueError("Supplied local PSFs are required for the fitted multiband preprocessing")
        from .filtering import match_gaussian_psf
        target = max(psfs)
        images = {band: match_gaussian_psf(im, target) for band, im in images.items()}
    features = arm_features(images, geometry, tuple(artifact["scales_sigma_arcsec"]))
    probability, raw = predict_feature_probabilities(artifact, features)
    field = np.full(features.valid.shape, np.nan, dtype=np.float32)
    field[features.valid] = probability
    binary = np.zeros(features.valid.shape, bool)
    binary[features.valid] = probability >= artifact["threshold"]
    return {"probability": field, "binary": binary, "valid": features.valid,
        "descriptors": {**features.descriptors, **probability_descriptors(probability, features, artifact["threshold"])},
        "metadata": {**features.metadata, "threshold": artifact["threshold"],
            "model": artifact.get("selected_model"),
            "probability_meaning": artifact.get("probability_meaning", "human annotation-agreement score under training domain; not physical arm certainty"),
            "limitations": artifact.get("limitations", [])}}
