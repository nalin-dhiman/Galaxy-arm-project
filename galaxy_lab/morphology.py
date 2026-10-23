"""Calibrated-image geometry and logarithmic-spiral Fourier measurements.

The native FITS pixels are used; display stretches never enter the measurement.
Fourier modes describe periodic structure, not separately identified arm segments.
Method: Savchenko & Reshetnikov (2013), MNRAS 436, 1074,
https://academic.oup.com/mnras/article/436/2/1074/2907733 .
Angles here are in the image plane, from +x toward +y, unless explicitly named
``pa_sky_deg`` (north through east). A supplied distance is an external input,
never an inference from these morphological measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass
class FitsImage:
    data: np.ndarray
    metadata: dict[str, Any]


def _robust_location(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values)[np.isfinite(values)]
    if values.size < 10:
        raise ValueError("Fewer than ten valid background samples")
    for _ in range(4):
        med = float(np.median(values))
        sigma = float(1.4826 * np.median(np.abs(values - med)))
        if sigma <= 0:
            return med, 0.0
        keep = np.abs(values - med) < 3.5 * sigma
        if keep.all():
            break
        values = values[keep]
    return med, sigma


def estimate_background(data: np.ndarray) -> tuple[float, float]:
    """Sigma-clipped border estimate; extended galaxies can bias this high."""
    ny, nx = data.shape
    border = max(3, int(min(nx, ny) * .12))
    mask = np.ones(data.shape, dtype=bool)
    mask[border:-border, border:-border] = False
    return _robust_location(data[mask])


def load_fits(path: str | Path, hdu: int | str | None = None) -> FitsImage:
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    with fits.open(path, memmap=False) as hdus:
        if hdu is None:
            choices = [i for i, item in enumerate(hdus)
                       if item.data is not None and np.ndim(item.data) == 2]
            if not choices:
                raise ValueError("No two-dimensional image HDU")
            hdu = choices[0]
        item = hdus[hdu]
        data = np.asarray(item.data, dtype=float).copy()
        header = item.header.copy()
    if data.ndim != 2 or min(data.shape) < 24:
        raise ValueError("Need a two-dimensional image at least 24 pixels wide")
    metadata: dict[str, Any] = {
        "path": str(path), "hdu": hdu, "shape": list(data.shape),
        "bunit": header.get("BUNIT", "unspecified"),
        "filter": header.get("FILTER", header.get("BAND", header.get("BAND0", header.get("BANDS", "unspecified")))),
        "finite_fraction": float(np.isfinite(data).mean()),
    }
    try:
        wcs = WCS(header).celestial
        if wcs.has_celestial:
            scales = np.asarray(proj_plane_pixel_scales(wcs)) * 3600
            metadata["pixel_scales_arcsec"] = scales.tolist()
            metadata["pixel_scale_arcsec"] = float(np.sqrt(np.prod(scales)))
            metadata["wcs"] = wcs
    except (ValueError, TypeError):
        pass
    return FitsImage(data, metadata)


def sky_center_and_pa(metadata: dict, ra_deg: float, dec_deg: float,
                      pa_sky_deg: float | None = None) -> tuple[tuple[float, float], float | None]:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    wcs = metadata.get("wcs")
    if wcs is None:
        raise ValueError("RA/Dec center requires celestial WCS")
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    x, y = wcs.world_to_pixel(c)
    pa = None
    if pa_sky_deg is not None:
        end = c.directional_offset_by(pa_sky_deg * u.deg, 10 * u.arcsec)
        xe, ye = wcs.world_to_pixel(end)
        pa = float(np.degrees(np.arctan2(ye - y, xe - x)))
    return (float(x), float(y)), pa


def moment_geometry(data: np.ndarray, center_xy: tuple[float, float] | None = None,
                    background: float | None = None,
                    noise: float | None = None) -> dict[str, float]:
    """Central-aperture second moments; light-shape estimate, not fitted disk."""
    if background is None or noise is None:
        bg, ns = estimate_background(data)
        background = bg if background is None else background
        noise = ns if noise is None else noise
    valid = np.isfinite(data)
    image = np.where(valid, data - background, 0)
    smooth = gaussian_filter(image, 1.5)
    yy, xx = np.indices(data.shape)
    ny, nx = data.shape
    center = np.array(center_xy if center_xy is not None else ((nx - 1) / 2, (ny - 1) / 2), float)
    # Restrict the moment aperture so stars near the cutout boundary cannot
    # dominate. No automatic source-deblending claim is made.
    for _ in range(3):
        rr = np.hypot(xx - center[0], yy - center[1])
        weight = np.clip(smooth - max(2 * noise, 0), 0, None)
        weight *= (rr < .40 * min(nx, ny)) & valid
        if weight.sum() <= 0:
            raise ValueError("No central positive signal above background")
        if center_xy is None:
            center = np.array([(weight * xx).sum(), (weight * yy).sum()]) / weight.sum()
    dx, dy = xx - center[0], yy - center[1]
    covariance = np.array([[(weight * dx * dx).sum(), (weight * dx * dy).sum()],
                           [(weight * dx * dy).sum(), (weight * dy * dy).sum()]]) / weight.sum()
    eigval, eigvec = np.linalg.eigh(covariance)
    q = float(np.sqrt(max(eigval[0], 0) / max(eigval[1], 1e-15)))
    major = eigvec[:, 1]
    pa = float(np.degrees(np.arctan2(major[1], major[0])) % 180)
    return {"center_x_px": float(center[0]), "center_y_px": float(center[1]),
            "axis_ratio_moment": q, "pa_image_deg_moment": pa,
            "sigma_major_px": float(np.sqrt(eigval[1]))}


def inclination_from_q(q: float, intrinsic_thickness: float = 0.) -> float:
    if not 0 <= intrinsic_thickness < 1 or not 0 < q <= 1:
        raise ValueError("Require 0 < q <= 1 and 0 <= intrinsic thickness < 1")
    cos2 = np.clip((q * q - intrinsic_thickness ** 2) / (1 - intrinsic_thickness ** 2), 0, 1)
    return float(np.degrees(np.arccos(np.sqrt(cos2))))


def projected_pair_separation(ra1_deg: float, dec1_deg: float,
                              ra2_deg: float, dec2_deg: float,
                              distance_mpc: float | None = None) -> dict:
    """Sky separation; physical transverse separation needs external distance.

    The optional small-angle conversion assumes both sources are associated
    with the same distance plane. It says nothing about line-of-sight spacing.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    if not np.isfinite([ra1_deg, dec1_deg, ra2_deg, dec2_deg]).all():
        raise ValueError("Sky coordinates must be finite")
    angle = SkyCoord(ra1_deg * u.deg, dec1_deg * u.deg).separation(
        SkyCoord(ra2_deg * u.deg, dec2_deg * u.deg))
    result = {"angular_separation_arcsec": float(angle.arcsec),
              "physical_assumption": "external common angular-diameter distance; unknown line-of-sight separation"}
    if distance_mpc is not None:
        if not np.isfinite(distance_mpc) or distance_mpc <= 0:
            raise ValueError("External distance must be finite and positive")
        result["external_distance_mpc"] = float(distance_mpc)
        result["projected_separation_kpc_small_angle"] = float(angle.radian * distance_mpc * 1000)
    return result


def elliptical_radius(shape: tuple[int, int], center_xy: tuple[float, float],
                      pa_deg: float, inclination_deg: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    dx, dy = xx - center_xy[0], yy - center_xy[1]
    c, s = np.cos(np.radians(pa_deg)), np.sin(np.radians(pa_deg))
    return np.hypot(c * dx + s * dy, (-s * dx + c * dy) / np.cos(np.radians(inclination_deg)))


def logpolar_fourier(data: np.ndarray, center_xy: tuple[float, float], pa_deg: float,
                     inclination_deg: float, r_inner_px: float, r_outer_px: float,
                     modes: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
                     n_radius: int = 160, n_theta: int = 360,
                     pitch_range_deg: tuple[float, float] = (5., 60.)) -> dict:
    """Fit mean log-spiral modes in a deprojected annulus.

    The azimuthal mean is removed ring by ring and residuals divided by each
    ring's mean positive brightness. Thus this measures relative arm contrast
    rather than allowing the bright bulge/inner disk to dominate the integral.
    A smooth radial taper suppresses finite-annulus sidelobes. Both winding
    senses are searched, and p=0 (bar/radial mode) is reported separately.
    """
    if not (0 <= inclination_deg < 80):
        raise ValueError("Deprojection requires 0 <= inclination < 80 degrees")
    if not 2 <= r_inner_px < r_outer_px:
        raise ValueError("Require 2 <= inner radius < outer radius")
    uu = np.linspace(np.log(r_inner_px), np.log(r_outer_px), n_radius)
    theta = np.arange(n_theta) * (2 * np.pi / n_theta)
    radius = np.exp(uu)
    X = radius[:, None] * np.cos(theta)
    Y = radius[:, None] * np.sin(theta) * np.cos(np.radians(inclination_deg))
    c, s = np.cos(np.radians(pa_deg)), np.sin(np.radians(pa_deg))
    xp, yp = center_xy[0] + c * X - s * Y, center_xy[1] + s * X + c * Y
    valid = np.isfinite(data)
    sampled = map_coordinates(np.where(valid, data, 0), [yp, xp], order=1, mode="constant", cval=0)
    support = map_coordinates(valid.astype(float), [yp, xp], order=1, mode="constant", cval=0)
    good = support > .99
    coverage = float(good.mean())
    if coverage < .92:
        raise ValueError(f"Insufficient annulus coverage ({coverage:.1%})")
    sampled = np.where(good, sampled, np.nan)
    means = np.nanmean(sampled, axis=1)
    scale_floor = max(float(np.nanmax(np.abs(means))) * .01, 1e-15)
    relative = (sampled - means[:, None]) / np.maximum(means[:, None], scale_floor)
    relative = np.where(np.isfinite(relative), relative, 0)
    taper = np.hanning(n_radius)
    positive = np.linspace(pitch_range_deg[0], pitch_range_deg[1], 551)
    pitches = np.concatenate((-positive[::-1], positive))
    results = []
    spectra = {}
    for m in modes:
        angular = np.mean(relative * np.exp(-1j * m * theta), axis=1)
        p = -m / np.tan(np.radians(pitches))
        amplitudes = np.abs(np.exp(-1j * p[:, None] * uu) @ (angular * taper)) / taper.sum()
        peak = int(np.argmax(amplitudes))
        bar_amplitude = float(np.abs(np.sum(angular * taper)) / taper.sum())
        results.append({"mode_m": int(m), "pitch_signed_deg": float(pitches[peak]),
                        "pitch_abs_deg": float(abs(pitches[peak])),
                        "amplitude": float(amplitudes[peak]),
                        "radial_bar_amplitude": bar_amplitude,
                        "p_log_wavenumber": float(p[peak]),
                        "search_boundary": bool(peak in (0, len(pitches) - 1) or abs(pitches[peak]) <= pitch_range_deg[0] + .11)})
        spectra[str(m)] = amplitudes
    best = max(results, key=lambda item: item["amplitude"])
    radial = []
    for indices in np.array_split(np.arange(n_radius), 6):
        local = np.nanmean(np.abs(np.fft.rfft(relative[indices], axis=1) / n_theta)[:, list(modes)], axis=0)
        m = int(modes[int(np.argmax(local))])
        radial.append({"radius_px": float(np.exp(np.mean(uu[indices]))), "mode_m": m,
                       "periodic_azimuth_spacing_deg": 360. / m,
                       "mode_amplitude": float(np.max(local))})
    return {**best, "coverage_fraction": coverage, "modes": results,
            "radial_mode_spacing": radial,
            "diagnostics": {"log_radius": uu, "theta": theta, "polar_relative": relative,
                            "pitch_grid_deg": pitches, "spectra": spectra}}


def measure_image(data: np.ndarray, *, pixel_scale_arcsec: float | None = None,
                  center_xy: tuple[float, float] | None = None,
                  pa_deg: float | None = None, inclination_deg: float | None = None,
                  intrinsic_thickness: float = 0., r_inner_px: float | None = None,
                  r_outer_px: float | None = None, distance_mpc: float | None = None,
                  sensitivity: bool = True) -> dict:
    """Return measurements and explicit quality flags, with no fitted distance."""
    data = np.asarray(data, float)
    if data.ndim != 2 or np.isfinite(data).sum() < 100:
        raise ValueError("Need at least 100 finite image pixels")
    if distance_mpc is not None and (not np.isfinite(distance_mpc) or distance_mpc <= 0):
        raise ValueError("External angular-diameter distance must be finite and positive")
    background, noise = estimate_background(data)
    geometry = moment_geometry(data, center_xy, background, noise)
    center = (geometry["center_x_px"], geometry["center_y_px"])
    pa = geometry["pa_image_deg_moment"] if pa_deg is None else float(pa_deg)
    q = geometry["axis_ratio_moment"]
    inc = inclination_from_q(q, intrinsic_thickness) if inclination_deg is None else float(inclination_deg)
    if not 0 <= inc < 80:
        raise ValueError("Inferred/supplied inclination >= 80 degrees; unstable deprojection")
    clean = data - background
    rr = elliptical_radius(data.shape, center, pa, inc)
    ny, nx = data.shape
    edge = float(min(center[0], nx - 1 - center[0], center[1], ny - 1 - center[1]))
    if edge < 12:
        raise ValueError("Galaxy center is outside or too near the image boundary")
    # Curve of growth within conservative complete-aperture extent.
    good = np.isfinite(clean) & (rr <= .94 * edge)
    radii, flux = rr[good].ravel(), np.clip(clean[good], 0, None).ravel()
    order = np.argsort(radii)
    cumulative = np.cumsum(flux[order])
    if cumulative[-1] <= 0:
        raise ValueError("Nonpositive galaxy flux")
    r50 = float(radii[order][np.searchsorted(cumulative, .5 * cumulative[-1])])
    r90 = float(radii[order][np.searchsorted(cumulative, .9 * cumulative[-1])])
    outer = float(r_outer_px if r_outer_px is not None else min(r90, .90 * edge))
    inner = float(r_inner_px if r_inner_px is not None else max(6., .25 * outer))
    spiral = logpolar_fourier(clean, center, pa, inc, inner, outer)
    flags = []
    if pa_deg is None or inclination_deg is None:
        flags.append("MOMENT_GEOMETRY_UNCALIBRATED: disk/bar/arms/foreground sources can bias shape")
    if noise == 0:
        flags.append("ZERO_BACKGROUND_SCATTER: synthetic/quantized/oversmoothed image or no noise estimate")
    if q > .9 and pa_deg is None:
        flags.append("NEAR_FACE_ON_PA_WEAKLY_CONSTRAINED")
    if noise > 0 and np.nanmedian(clean[(rr > inner) & (rr < outer)]) < 3 * noise:
        flags.append("LOW_ANNULUS_SURFACE_BRIGHTNESS")
    if spiral["amplitude"] < .03:
        flags.append("WEAK_PERIODIC_STRUCTURE")
    if spiral["coverage_fraction"] < .995:
        flags.append("MISSING_ANNULUS_PIXELS_CAN_ALIAS_FOURIER_MODES")
    if spiral["radial_bar_amplitude"] >= .8 * spiral["amplitude"]:
        flags.append("BAR_OR_NONWINDING_STRUCTURE_COMPETES_WITH_SPIRAL")
    if spiral["search_boundary"]:
        flags.append("PITCH_AT_SEARCH_BOUNDARY")
    if inc > 65:
        flags.append("HIGH_INCLINATION_SENSITIVE_TO_DEPROJECTION")
    trials = []
    if sensitivity:
        # These deterministic perturbations are sensitivity, not statistical
        # confidence intervals. They do not include all source/sky/PSF errors.
        trial_parameters = [("annulus_inner_plus15pct", pa, inc, inner * 1.15, outer),
                            ("annulus_outer_minus15pct", pa, inc, inner, outer * .85),
                            ("pa_minus5deg", pa - 5, inc, inner, outer),
                            ("pa_plus5deg", pa + 5, inc, inner, outer),
                            ("inclination_minus5deg", pa, max(0, inc - 5), inner, outer),
                            ("inclination_plus5deg", pa, min(79., inc + 5), inner, outer)]
        for label, tp, ti, rin, rout in trial_parameters:
            try:
                result = logpolar_fourier(clean, center, tp, ti, rin, rout, n_radius=128, n_theta=256)
                trials.append({"perturbation": label, "mode_m": result["mode_m"],
                               "pitch_abs_deg": result["pitch_abs_deg"], "status": "ok"})
            except ValueError as error:
                trials.append({"perturbation": label, "status": "failed", "reason": str(error)})
        successful = [t for t in trials if t["status"] == "ok"]
        same = [t["pitch_abs_deg"] for t in successful if t["mode_m"] == spiral["mode_m"]]
        if len(same) != len(successful):
            flags.append("DOMINANT_MODE_UNSTABLE_UNDER_GEOMETRY_OR_ANNULUS")
        if same and np.ptp(same) > 5:
            flags.append("PITCH_SENSITIVE_OVER_5DEG")
        spiral["sensitivity_pitch_min_deg"] = float(min(same)) if same else None
        spiral["sensitivity_pitch_max_deg"] = float(max(same)) if same else None
    result = {"status": "measured_with_flags" if flags else "measured", "flags": flags,
              "background": background, "background_sigma": noise,
              "finite_fraction": float(np.isfinite(data).mean()),
              **geometry, "pa_image_deg_used": pa, "inclination_deg_used": inc,
              "intrinsic_thickness_assumed": intrinsic_thickness,
              "geometry_source": {"center": "supplied" if center_xy is not None else "central moments",
                                  "pa": "supplied" if pa_deg is not None else "central moments",
                                  "inclination": "supplied" if inclination_deg is not None else "central moments"},
              "r50_aperture_px": r50, "r90_aperture_px": r90,
              "curve_of_growth_max_radius_px": .94 * edge,
              "r_inner_px": inner, "r_outer_px": outer,
              "spiral": spiral, "sensitivity_trials": trials,
              "distance_inference": "not identifiable from geometry alone",
              "distance_mpc_external": distance_mpc,
              "limitations": ["No foreground-source deblending or bar/bulge fit",
                              "Border sky may include galaxy light; aperture radii are not total-light radii",
                              "Fourier mode spacing is periodicity, not traced arm or galaxy-pair separation",
                              "Sensitivity range is not a confidence interval",
                              "Quality thresholds are heuristics, not calibrated detection significance"]}
    if pixel_scale_arcsec is not None:
        if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
            raise ValueError("Pixel scale must be finite and positive")
        result["pixel_scale_arcsec"] = float(pixel_scale_arcsec)
        result["r50_aperture_arcsec"] = float(r50 * pixel_scale_arcsec)
        result["r90_aperture_arcsec"] = float(r90 * pixel_scale_arcsec)
        for radial in spiral["radial_mode_spacing"]:
            radial["radius_arcsec"] = radial["radius_px"] * pixel_scale_arcsec
            radial["deprojected_periodic_arc_length_arcsec"] = radial["radius_arcsec"] * 2 * np.pi / radial["mode_m"]
        if distance_mpc is not None:
            if distance_mpc <= 0:
                raise ValueError("External angular-diameter distance must be positive")
            result["r50_kpc_at_external_distance"] = result["r50_aperture_arcsec"] * distance_mpc * 1000 / 206264.806247
    return result


def synthetic_spiral(size: int = 256, pitch_deg: float = 22., mode_m: int = 2,
                     inclination_deg: float = 35., pa_deg: float = 25.,
                     psf_sigma_px: float = 1., noise_sigma: float = .02,
                     seed: int = 1) -> tuple[np.ndarray, dict]:
    """Controlled injected light field, with known geometry and pitch.

    No claim that this simple field reproduces clumps/dust/warps or real surveys.
    """
    center = ((size - 1) / 2, (size - 1) / 2)
    yy, xx = np.indices((size, size))
    c, s = np.cos(np.radians(pa_deg)), np.sin(np.radians(pa_deg))
    dx, dy = xx - center[0], yy - center[1]
    X, Y = c * dx + s * dy, (-s * dx + c * dy) / np.cos(np.radians(inclination_deg))
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    phase = mode_m * (theta - np.log(np.maximum(r, 1) / (size * .1)) / np.tan(np.radians(pitch_deg)))
    disk = np.exp(-r / (size * .20))
    window = (1 - np.exp(-(r / (size * .07)) ** 4)) * np.exp(-(r / (size * .47)) ** 8)
    image = disk * (1 + .55 * window * np.cos(phase)) + .6 * np.exp(-.5 * (r / (size * .025)) ** 2)
    image = gaussian_filter(image, psf_sigma_px)
    image += .2 + np.random.default_rng(seed).normal(0, noise_sigma, image.shape)
    return image, {"center_xy": center, "pa_deg": pa_deg, "inclination_deg": inclination_deg,
                   "r_inner_px": size * .11, "r_outer_px": size * .35,
                   "pitch_deg": pitch_deg, "mode_m": mode_m}
