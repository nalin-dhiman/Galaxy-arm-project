"""Angular-scale image filters and masked validation, separate from photometry.

A filter score highlights a type of structure; it is never an arm identification,
flux estimate, distance estimate, or evidence of a material viscosity. Rendered
RGB images may be analysed as display data, but cannot become calibrated science
bands by applying channel weights. No label mask enters any image filter.

References: scipy.ndimage filter documentation;
https://docs.scipy.org/doc/scipy/reference/ndimage.html
https://scikit-image.org/docs/stable/api/skimage.exposure.html
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.ndimage import (gaussian_filter, map_coordinates, maximum_filter,
                           grey_opening, distance_transform_edt, convolve1d,
                           binary_propagation)
from scipy.optimize import least_squares
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class ScienceImage:
    data: np.ndarray
    pixel_scale_arcsec: float
    unit: str = "unspecified"
    mask: np.ndarray | None = None  # True = invalid/excluded
    variance: np.ndarray | None = None
    wcs: Any = None
    psf_fwhm_arcsec: float | None = None
    band: str = "unknown"
    provenance: str = "science"  # 'rendered_display' for RGB volunteer images
    pixel_value_kind: str = "surface_brightness"  # or flux_per_pixel
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2 or min(self.data.shape) < 8:
            raise ValueError("A two-dimensional image at least 8 pixels wide is required")
        if not np.isfinite(self.pixel_scale_arcsec) or self.pixel_scale_arcsec <= 0:
            raise ValueError("Pixel scale must be finite and positive")
        for name in ("mask", "variance"):
            value = getattr(self, name)
            if value is not None and np.shape(value) != self.data.shape:
                raise ValueError(f"{name} shape differs from image")
        if self.pixel_value_kind not in ("surface_brightness", "flux_per_pixel"):
            raise ValueError("Declare surface_brightness or flux_per_pixel")
        if self.psf_fwhm_arcsec is not None and (not np.isfinite(self.psf_fwhm_arcsec) or self.psf_fwhm_arcsec <= 0):
            raise ValueError("Supplied PSF FWHM must be positive")

    @property
    def valid(self):
        result = np.isfinite(self.data)
        if self.mask is not None:
            result &= ~np.asarray(self.mask, dtype=bool)
        if self.variance is not None:
            result &= np.isfinite(self.variance) & (self.variance > 0)
        return result


@dataclass(frozen=True)
class DiskGeometry:
    center_xy: tuple[float, float]
    pa_deg: float
    inclination_deg: float
    r_inner_arcsec: float
    r_outer_arcsec: float
    source: str = "external"

    def __post_init__(self):
        if not np.isfinite([*self.center_xy, self.pa_deg, self.inclination_deg,
                            self.r_inner_arcsec, self.r_outer_arcsec]).all():
            raise ValueError("Geometry must be finite")
        if not 0 <= self.inclination_deg < 80:
            raise ValueError("Inclination must lie in [0,80) degrees")
        if not 0 <= self.r_inner_arcsec < self.r_outer_arcsec:
            raise ValueError("Require 0 <= inner < outer radius")


@dataclass
class FilterMap:
    score: np.ndarray
    valid: np.ndarray
    metadata: dict


def radius_map(shape, geometry: DiskGeometry, pixel_scale_arcsec: float):
    yy, xx = np.indices(shape)
    dx, dy = xx - geometry.center_xy[0], yy - geometry.center_xy[1]
    angle = np.radians(geometry.pa_deg)
    xp = dx * np.cos(angle) + dy * np.sin(angle)
    yp = -dx * np.sin(angle) + dy * np.cos(angle)
    return np.hypot(xp, yp / np.cos(np.radians(geometry.inclination_deg))) * pixel_scale_arcsec


def annulus_mask(image: ScienceImage, geometry: DiskGeometry):
    rr = radius_map(image.data.shape, geometry, image.pixel_scale_arcsec)
    return image.valid & (rr >= geometry.r_inner_arcsec) & (rr <= geometry.r_outer_arcsec)


def robust_background(image: ScienceImage, *, border_fraction: float = .12):
    """Clipped border median/MAD; extended source contamination remains possible."""
    if not 0 < border_fraction < .5:
        raise ValueError("Border fraction must lie between zero and 0.5")
    width = max(2, int(min(image.data.shape) * border_fraction))
    border = np.ones(image.data.shape, dtype=bool)
    border[width:-width, width:-width] = False
    values = image.data[border & image.valid]
    if values.size < 20:
        raise ValueError("Insufficient valid border pixels")
    for _ in range(5):
        med = float(np.median(values))
        noise = float(1.4826 * np.median(np.abs(values - med)))
        if noise <= 0:
            break
        keep = abs(values - med) < 3.5 * noise
        if keep.all():
            break
        values = values[keep]
    return med, noise


def _nearest_fill(data, valid):
    if not np.any(valid):
        raise ValueError("Image has no valid pixels")
    if valid.all():
        return np.asarray(data, float).copy()
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(data, float)[tuple(indices)]


def masked_gaussian(data, valid, sigma):
    """Normalized convolution; excluded values cannot contribute to output."""
    weight = gaussian_filter(valid.astype(float), sigma, mode="constant", cval=0)
    numer = gaussian_filter(np.where(valid, data, 0), sigma, mode="constant", cval=0)
    result = np.full(data.shape, np.nan)
    np.divide(numer, weight, out=result, where=weight > 1e-8)
    return result, weight


def radial_residual(image: ScienceImage, geometry: DiskGeometry):
    """Subtract a median elliptical radial profile, independent of arm labels."""
    rr = radius_map(image.data.shape, geometry, image.pixel_scale_arcsec)
    radial_bin = np.floor(rr / image.pixel_scale_arcsec).astype(int)
    max_bin = int(radial_bin.max()) + 1
    profile = np.full(max_bin, np.nan)
    for k in range(max_bin):
        values = image.data[image.valid & (radial_bin == k)]
        if values.size >= 4:
            profile[k] = np.median(values)
    valid_bins = np.flatnonzero(np.isfinite(profile))
    if valid_bins.size < 3:
        raise ValueError("Insufficient valid radial samples")
    profile = np.interp(np.arange(max_bin), valid_bins, profile[valid_bins])
    profile = gaussian_filter(profile, 1.)
    return image.data - profile[radial_bin], profile


def filter_bank(image: ScienceImage, geometry: DiskGeometry,
                scales_arcsec: tuple[float, float, float] = (.4, .8, 1.6)) -> dict[str, FilterMap]:
    """Fixed ten-map bank, with scales in sky arcseconds (Gaussian sigmas).

    The three scales and all thresholds must be fixed on development data before
    evaluating held-out galaxies. Derivative maps use scale normalization; all
    maps are structural scores, not calibrated fluxes. A common valid mask is
    used for every map. Scores near holes within four largest sigmas are excluded
    because interpolation derivatives and morphology there are unreliable.
    """
    scales = np.asarray(scales_arcsec, dtype=float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or not np.all(scales > 0) or not np.all(np.diff(scales) > 0):
        raise ValueError("Supply three positive increasing angular scales")
    sigmas = scales / image.pixel_scale_arcsec
    if sigmas[0] < .6:
        raise ValueError("Finest angular sigma is undersampled (<0.6 pixel)")
    background, noise = robust_background(image)
    residual, _ = radial_residual(image, geometry)
    filled = _nearest_fill(residual, image.valid)
    smooth = [gaussian_filter(filled, s, mode="reflect") for s in sigmas]
    hyy = gaussian_filter(filled, sigmas[1], order=(2, 0)) * sigmas[1] ** 2
    hxx = gaussian_filter(filled, sigmas[1], order=(0, 2)) * sigmas[1] ** 2
    hxy = gaussian_filter(filled, sigmas[1], order=(1, 1)) * sigmas[1] ** 2
    # Bright ridge response suppresses blobs with similarly negative eigenvalues.
    root = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
    small, large = (hxx + hyy - root) / 2, (hxx + hyy + root) / 2
    swap = abs(small) > abs(large)
    along = np.where(swap, large, small)
    across = np.where(swap, small, large)
    ridge = np.maximum(-across, 0) * np.exp(-.5 * (along / np.maximum(abs(across), 1e-15) / .5) ** 2)
    # Circular footprint radius is 2 medium sigmas, specified by angular scales.
    rad = max(1, int(np.ceil(2 * sigmas[1])))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    footprint = xx * xx + yy * yy <= rad * rad
    top_hat = filled - grey_opening(filled, footprint=footprint)
    gx = gaussian_filter(filled, sigmas[0], order=(0, 1)) * sigmas[0]
    gy = gaussian_filter(filled, sigmas[0], order=(1, 0)) * sigmas[0]
    edge = np.hypot(gx, gy)
    # Hysteresis edges are binary display/segmentation candidates, never arm IDs.
    edge_noise = max(float(np.median(edge[image.valid])) / 1.1774, noise * .05, 1e-12)
    high, low = edge > 4 * edge_noise, edge > 2 * edge_noise
    hysteresis = binary_propagation(high, mask=low).astype(float)
    maps = {
        "native_intensity": image.data - background,
        "radial_residual": residual,
        "dog_fine": smooth[0] - smooth[1],
        "dog_coarse": smooth[1] - smooth[2],
        "unsharp": filled - smooth[2],
        "negative_log": -(hxx + hyy),
        "bright_ridge": ridge,
        "white_tophat": top_hat,
        "gradient_edges": edge,
        "hysteresis_binary": hysteresis,
    }
    common = annulus_mask(image, geometry)
    before_guard = int(common.sum())
    rr = radius_map(image.data.shape, geometry, image.pixel_scale_arcsec)
    geometric_annulus = int(((rr >= geometry.r_inner_arcsec) & (rr <= geometry.r_outer_arcsec)).sum())
    # Do not erode whole image at borders: the annulus must stay inside these.
    safe = image.valid.copy()
    if not safe.all():
        safe &= distance_transform_edt(safe) > 4 * sigmas[-1]
    border = int(np.ceil(4 * sigmas[-1]))
    safe[:border] = safe[-border:] = False
    safe[:, :border] = safe[:, -border:] = False
    common &= safe
    meta = {"scales_sigma_arcsec": scales.tolist(), "input_unit": image.unit,
            "input_band": image.band, "input_provenance": image.provenance,
            "geometry_source": geometry.source,
            "background": background, "border_noise_mad": noise,
            "interpretation": "structure ranking score; not photometry or confirmed spiral arms",
            "common_valid_pixels": int(common.sum()),
            "annulus_geometric_pixels": geometric_annulus,
            "annulus_input_valid_pixels": before_guard,
            "guard_sigma_largest_scale": 4.,
            "coverage_of_geometric_annulus": float(common.sum() / geometric_annulus) if geometric_annulus else 0.,
            "guard_retained_fraction": float(common.sum() / before_guard) if before_guard else 0.}
    return {name: FilterMap(np.where(common, score, np.nan), common.copy(),
                           {**meta, "filter": name}) for name, score in maps.items()}


def display_grayscale(rgb: np.ndarray, channel_axis: int = -1):
    """Rendered RGB luminance for display analysis ONLY, no band/flux recovery."""
    color = np.moveaxis(np.asarray(rgb, float), channel_axis, -1)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError("Require exactly three rendered RGB channels")
    return np.sum(color * np.array([.2126, .7152, .0722]), axis=-1)


def display_transform(data, method="asinh", *, percentiles=(1., 99.5),
                      binary_threshold=.5, clahe_clip=.02):
    """Independent display copy scaled to [0,1]; never used as calibrated flux."""
    source = np.asarray(data, float)
    valid = np.isfinite(source)
    if not valid.any():
        raise ValueError("No finite display pixels")
    lo, hi = np.percentile(source[valid], percentiles)
    span = max(float(hi - lo), np.finfo(float).eps)
    scaled = np.clip((np.where(valid, source, lo) - lo) / span, 0, 1)
    if method == "asinh":
        out = np.arcsinh(10 * scaled) / np.arcsinh(10)
    elif method == "linear":
        out = scaled
    elif method == "binary":
        if not 0 <= binary_threshold <= 1:
            raise ValueError("Threshold must be in [0,1]")
        out = (scaled >= binary_threshold).astype(float)
    elif method == "clahe":
        from skimage.exposure import equalize_adapthist
        out = equalize_adapthist(scaled, clip_limit=clahe_clip)
    else:
        raise ValueError("Choose linear, asinh, clahe or binary")
    return np.where(valid, out, np.nan)


def evaluate_score(score, truth, valid, *, threshold: float | None = None):
    """Pixel AP against independent binary labels within a predeclared region.

    Pixels are correlated; confidence intervals must resample whole galaxies,
    not pixels. Threshold-dependent precision/recall require a development-set
    threshold. With no threshold supplied they are intentionally omitted.
    """
    score, truth, valid = np.asarray(score), np.asarray(truth, bool), np.asarray(valid, bool)
    if score.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError("Score, truth and valid shapes must agree")
    good = valid & np.isfinite(score)
    y, s = truth[good], score[good]
    n, positives = len(y), int(y.sum())
    result = {"valid_pixels": n, "positive_pixels": positives,
              "positive_fraction": float(positives / n) if n else None,
              "average_precision": None, "roc_auc": None}
    if 0 < positives < n:
        result["average_precision"] = float(average_precision_score(y, s))
        result["roc_auc"] = float(roc_auc_score(y, s))
    if threshold is not None and n:
        pred = s >= threshold
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        result.update(threshold=float(threshold), true_positive=tp, false_positive=fp,
                      false_negative=fn, precision=tp / (tp + fp) if tp + fp else 0.,
                      recall=tp / positives if positives else None)
    return result


def foreground_point_mask(image: ScienceImage, center_xy, *, psf_fwhm_arcsec: float,
                          exclude_center_arcsec: float = 3., threshold_sigma=7.,
                          manual_mask=None):
    """Conservative PSF-like compact peaks; knots/nuclei can remain ambiguous.

    The target nucleus is protected. This heuristic is NOT star classification.
    Use independently checked catalog/manual masks for scientific conclusions.
    Returns mask and accepted peak records; no human arm labels are consulted.
    """
    sigma = psf_fwhm_arcsec / image.pixel_scale_arcsec / 2.354820045
    if sigma < .6:
        raise ValueError("PSF undersampled for compact-source detection")
    background, noise = robust_background(image)
    filled = _nearest_fill(image.data - background, image.valid)
    residual = filled - gaussian_filter(filled, max(4 * sigma, 4.))
    matched = gaussian_filter(residual, sigma)
    # Gaussian smoothing reduces white background noise by sqrt(4*pi*sigma²).
    detection_noise = max(noise / (2 * np.sqrt(np.pi) * sigma), 1e-12)
    peaks = (matched == maximum_filter(matched, size=max(3, int(2 * sigma + 1)))) & (matched > threshold_sigma * detection_noise) & image.valid
    yy, xx = np.indices(image.data.shape)
    peaks &= np.hypot(xx - center_xy[0], yy - center_xy[1]) * image.pixel_scale_arcsec > exclude_center_arcsec
    mask = np.zeros(image.data.shape, bool)
    accepted = []
    for y, x in zip(*np.where(peaks)):
        rr2 = (xx - x) ** 2 + (yy - y) ** 2
        local = rr2 <= (2.5 * sigma) ** 2
        weight = np.maximum(residual, 0) * local
        if weight.sum() <= 0:
            continue
        cov = np.array([[(weight * (xx - x) ** 2).sum(), (weight * (xx - x) * (yy - y)).sum()],
                        [(weight * (xx - x) * (yy - y)).sum(), (weight * (yy - y) ** 2).sum()]]) / weight.sum()
        eig = np.linalg.eigvalsh(cov)
        # Loose compactness range for finite truncated aperture and noisy peaks.
        if eig[0] > .45 * sigma ** 2 and eig[1] < 1.8 * sigma ** 2 and eig[0] / eig[1] > .65:
            mask |= rr2 <= (3.5 * sigma) ** 2
            accepted.append({"x_px": int(x), "y_px": int(y), "major_sigma_px": float(np.sqrt(eig[1])),
                             "minor_sigma_px": float(np.sqrt(eig[0]))})
    if manual_mask is not None:
        if np.shape(manual_mask) != image.data.shape:
            raise ValueError("Manual mask shape differs")
        mask |= np.asarray(manual_mask, bool)
    return mask, accepted


def match_gaussian_psf(image: ScienceImage, target_fwhm_arcsec: float):
    """Broaden a supplied circular Gaussian PSF, propagating diagonal variance.

    Covariance introduced by smoothing is recorded but not represented by the
    variance image. Non-Gaussian spatially varying PSFs need measured kernels.
    """
    if image.provenance != "science":
        raise ValueError("PSF matching requires native science pixels")
    current = image.psf_fwhm_arcsec
    if current is None:
        raise ValueError("A measured/supplied input PSF is required")
    if target_fwhm_arcsec < current:
        raise ValueError("PSF matching cannot sharpen/deconvolve")
    sigma = np.sqrt(max(target_fwhm_arcsec ** 2 - current ** 2, 0)) / 2.354820045 / image.pixel_scale_arcsec
    if sigma < 1e-8:
        return ScienceImage(**{**image.__dict__, "data": image.data.copy(), "metadata": dict(image.metadata)})
    data, support = masked_gaussian(image.data, image.valid, sigma)
    variance = None
    if image.variance is not None:
        radius = int(4 * sigma + .5)
        coordinate = np.arange(-radius, radius + 1)
        kernel = np.exp(-.5 * (coordinate / sigma) ** 2)
        kernel /= kernel.sum()
        variance = np.where(image.valid, image.variance, 0)
        for axis in (0, 1):
            variance = convolve1d(variance, kernel ** 2, axis=axis, mode="constant", cval=0)
        variance = variance / np.maximum(support, 1e-12) ** 2
    mask = ~image.valid | (support < .99)
    return ScienceImage(**{**image.__dict__, "data": data, "mask": mask, "variance": variance,
        "psf_fwhm_arcsec": float(target_fwhm_arcsec), "metadata": {**image.metadata,
        "gaussian_psf_match": True, "correlated_noise": True,
        "psf_assumption": "circular stationary Gaussian; supplied input FWHM"}})


def reproject_to(image: ScienceImage, target_wcs, target_shape, target_pixel_scale_arcsec):
    """Bilinear WCS registration with squared-weight diagonal variance.

    Assumes square near-uniform pixel areas and smooth sky sampling. This is
    NOT an area-overlap/flux-conserving resampler: compact-source integrated
    flux can be biased, especially during downsampling. Prefer filtering native
    science grids and projecting annotation masks to those grids. Values at
    partial-coverage pixels are excluded. Reprojection introduces covariance;
    astrometric errors and full covariance are outside this approximation.
    The effective output PSF must be measured; its supplied input FWHM is not
    propagated as though interpolation had left the PSF unchanged.
    """
    if image.wcs is None or target_wcs is None:
        raise ValueError("Both celestial WCS solutions are required")
    if image.provenance != "science":
        raise ValueError("Reprojection API is reserved for native science pixels")
    yy, xx = np.indices(target_shape)
    world = target_wcs.pixel_to_world(xx, yy)
    xs, ys = image.wcs.world_to_pixel(world)
    finite = np.isfinite(xs) & np.isfinite(ys)
    x0 = np.floor(np.where(finite, xs, -2)).astype(int)
    y0 = np.floor(np.where(finite, ys, -2)).astype(int)
    dx, dy = xs - x0, ys - y0
    result = np.zeros(target_shape)
    variance = np.zeros(target_shape) if image.variance is not None else None
    support = np.zeros(target_shape)
    for ox, oy, weight in ((0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)),
                            (0, 1, (1 - dx) * dy), (1, 1, dx * dy)):
        xi, yi = x0 + ox, y0 + oy
        inside = finite & (xi >= 0) & (xi < image.data.shape[1]) & (yi >= 0) & (yi < image.data.shape[0])
        xc, yc = np.clip(xi, 0, image.data.shape[1] - 1), np.clip(yi, 0, image.data.shape[0] - 1)
        valid = inside & image.valid[yc, xc]
        w = np.where(valid, weight, 0.)
        support += w
        result += w * np.where(valid, image.data[yc, xc], 0.)
        if variance is not None:
            variance += w ** 2 * np.where(valid, image.variance[yc, xc], 0.)
    good = support >= .99999
    area_factor = (target_pixel_scale_arcsec / image.pixel_scale_arcsec) ** 2 if image.pixel_value_kind == "flux_per_pixel" else 1.
    result = np.where(good, result * area_factor, np.nan)
    if variance is not None:
        variance = np.where(good, variance * area_factor ** 2, np.nan)
    return ScienceImage(result, target_pixel_scale_arcsec, unit=image.unit, mask=~good,
        variance=variance, wcs=target_wcs, psf_fwhm_arcsec=None,
        band=image.band, provenance=image.provenance, pixel_value_kind=image.pixel_value_kind,
        metadata={**image.metadata, "reprojected_bilinear": True, "correlated_noise": True,
                  "pixel_area_approximation": "square uniform angular pixel area",
                  "photometry_validated": False,
                  "variance_approximation": "diagonal only; prior and introduced pixel covariances omitted",
                  "input_psf_fwhm_arcsec": image.psf_fwhm_arcsec,
                  "output_psf_warning": "bilinear interpolation changes the effective PSF; measure it before coadding"})


def inverse_variance_coadd(images: list[ScienceImage]):
    """Detection coadd of already PSF/WCS aligned science images in shared units.

    Mixed passbands produce a detection image, not a new physical passband.
    Requires known variances, declared units and equal supplied PSFs. Shared
    calibration errors/cross-band covariance are not represented here.
    """
    if not images:
        raise ValueError("No images to combine")
    first = images[0]
    if first.unit == "unspecified":
        raise ValueError("Coadd requires declared calibrated units")
    for image in images:
        if image.provenance != "science" or image.unit != first.unit or image.pixel_value_kind != first.pixel_value_kind:
            raise ValueError("Coadd requires science pixels in matching calibrated units/kinds")
        if image.variance is None or image.data.shape != first.data.shape or not np.isclose(image.pixel_scale_arcsec, first.pixel_scale_arcsec):
            raise ValueError("Images require variances and the same registered pixel grid")
        if image.psf_fwhm_arcsec is None or first.psf_fwhm_arcsec is None or not np.isclose(image.psf_fwhm_arcsec, first.psf_fwhm_arcsec):
            raise ValueError("PSFs must be supplied and matched before coadding")
        if image.wcs is None or first.wcs is None:
            raise ValueError("WCS registration must be declared")
        # Compare footprint and center world positions (not serialized headers).
        # Corners, edge midpoints and center detect rotation/shear that would
        # be invisible to samples lying only along the diagonal.
        x, y = np.meshgrid(np.linspace(0, image.data.shape[1] - 1, 3),
                           np.linspace(0, image.data.shape[0] - 1, 3))
        separation = first.wcs.pixel_to_world(x, y).separation(image.wcs.pixel_to_world(x, y)).arcsec
        if not np.isfinite(separation).all() or np.max(separation) > first.pixel_scale_arcsec * .01:
            raise ValueError("WCS grids differ: reproject before coadding")
    weights = [np.where(im.valid, 1 / np.where(im.valid, im.variance, 1.), 0.) for im in images]
    denominator = np.sum(weights, axis=0)
    numerator = np.sum([w * np.where(im.valid, im.data, 0.) for w, im in zip(weights, images)], axis=0)
    good = denominator > 0
    data = np.divide(numerator, denominator, out=np.full(first.data.shape, np.nan), where=good)
    variance = np.divide(1., denominator, out=np.full(first.data.shape, np.nan), where=good)
    return ScienceImage(**{**first.__dict__, "data": data, "variance": variance, "mask": ~good,
        "band": "+".join(im.band for im in images), "metadata": {**first.metadata,
        "coadd_kind": "inverse variance detection image", "bands": [im.band for im in images],
        "cross_band_covariance_included": False}})


def refine_outer_geometry(image: ScienceImage, initial: DiskGeometry,
                          q_bounds=(.3, 1.)):
    """Robust outer-isophote ellipse refinement with externally supplied center.

    Fits a local log-linear radial light profile plus ellipse parameters in the
    initial outer annulus. It is an image-shape fit, not kinematic inclination;
    bars, arms, dust and a non-exponential disk can bias it. Returns diagnostics.
    """
    rr = radius_map(image.data.shape, initial, image.pixel_scale_arcsec)
    bg, noise = robust_background(image)
    selected = image.valid & (rr > .55 * initial.r_outer_arcsec) & (rr < initial.r_outer_arcsec)
    signal = image.data - bg
    selected &= signal > max(2 * noise, np.nanmax(signal[image.valid]) * 1e-4)
    if selected.sum() < 100:
        raise ValueError("Too few outer-isophote pixels")
    yy, xx = np.indices(image.data.shape)
    dx, dy = (xx[selected] - initial.center_xy[0]), (yy[selected] - initial.center_xy[1])
    loglight = np.log(signal[selected])
    q0 = np.clip(np.cos(np.radians(initial.inclination_deg)), *q_bounds)
    # orientation is locally searched +/- 45 degrees; near face-on PA degeneracy remains.
    angle0 = np.radians(initial.pa_deg)
    scale = initial.r_outer_arcsec / image.pixel_scale_arcsec
    def fun(p):
        angle, q, a, b = p
        r = np.hypot(dx * np.cos(angle) + dy * np.sin(angle),
                     (-dx * np.sin(angle) + dy * np.cos(angle)) / q) / scale
        return a + b * r - loglight
    fit = least_squares(fun, [angle0, q0, np.median(loglight) + 1, -2],
        bounds=([angle0 - np.pi / 4, q_bounds[0], -np.inf, -30],
                [angle0 + np.pi / 4, q_bounds[1], np.inf, -.01]), loss="soft_l1", f_scale=.15)
    p = fit.x
    updated = DiskGeometry(initial.center_xy, float(np.degrees(p[0]) % 180),
                           float(np.degrees(np.arccos(p[1]))), initial.r_inner_arcsec,
                           initial.r_outer_arcsec, "outer light-isophote robust fit; not kinematic geometry")
    return updated, {"success": bool(fit.success), "outer_pixels": int(selected.sum()),
        "rms_log_residual": float(np.sqrt(np.mean(fun(p) ** 2))), "axis_ratio": float(p[1]),
        "warning": "spiral/dust/bar bias and face-on PA degeneracy; no calibrated geometry uncertainty"}
