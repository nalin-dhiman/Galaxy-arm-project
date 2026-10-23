"""Small command line interface for measurable quantities and model inspection."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()
                if key not in ("diagnostics", "wcs")}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hdu_selector(value):
    try:
        return int(value)
    except ValueError:
        return value


def _filter_command(args):
    """Export structural score maps, keeping display stretches separate."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from .filtering import (ScienceImage, DiskGeometry, filter_bank,
                            display_transform)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy_lab_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (args.center_x is None) != (args.center_y is None):
        raise ValueError("--center-x and --center-y must be supplied together")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty; existing results are preserved")
    with fits.open(args.image, memmap=False) as hdus:
        hdu = hdus[args.hdu]
        if hdu.data is None or np.ndim(hdu.data) != 2:
            raise ValueError("Selected science HDU must contain a two-dimensional image")
        data = np.asarray(hdu.data, float).copy()
        header = hdu.header.copy()
        variance = None
        if args.invvar_hdu is not None:
            invvar = np.asarray(hdus[args.invvar_hdu].data, dtype=float)
            if invvar.shape != data.shape:
                raise ValueError("Inverse-variance HDU shape must match the science image")
            good = np.isfinite(invvar) & (invvar > 0)
            variance = np.divide(1., invvar, out=np.full(data.shape, np.nan), where=good)
    try:
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            wcs = None
    except (ValueError, TypeError):
        wcs = None
    pixel_scale = args.pixel_scale_arcsec
    scales = None
    if wcs is not None:
        scales = np.asarray(proj_plane_pixel_scales(wcs), float) * 3600
    if pixel_scale is None:
        if scales is None or not np.isfinite(scales).all() or not np.all(scales > 0):
            raise ValueError("Supply --pixel-scale-arcsec when celestial WCS is unavailable")
        if max(scales) / min(scales) > 1.05:
            raise ValueError("WCS pixels differ in scale by more than 5%; this isotropic filter bank requires a suitable native grid")
        pixel_scale = float(np.sqrt(np.prod(scales)))
    ny, nx = data.shape
    center = (args.center_x, args.center_y) if args.center_x is not None else ((nx - 1) / 2, (ny - 1) / 2)
    if not (0 <= center[0] < nx and 0 <= center[1] < ny):
        raise ValueError("Galaxy center must lie inside the selected science image")
    unit = str(header.get("BUNIT", "unspecified")).strip() or "unspecified"
    band = args.band or str(header.get("FILTER", header.get("BAND", "unknown")))
    image = ScienceImage(data, pixel_scale, unit=unit, variance=variance, wcs=wcs,
        band=band, provenance="native_fits_units_unverified" if unit == "unspecified" else "native_fits",
        metadata={"calibration_verified": False})
    geometry_source = "user-supplied center/PA/inclination" if args.center_x is not None else "image-center default with specified PA/inclination"
    geometry = DiskGeometry(center, args.pa_deg, args.inclination_deg,
                            args.inner_arcsec, args.outer_arcsec, geometry_source)
    maps = filter_bank(image, geometry, tuple(args.scales_arcsec))
    common = next(iter(maps.values())).valid
    if common.sum() < 100:
        raise ValueError("Fewer than 100 common valid annulus pixels remain after source/edge guards; supply a suitable image/annulus/scales")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir
    export_header = wcs.to_header() if wcs is not None else fits.Header()
    export_header["PIXSCALE"] = (pixel_scale, "arcsec/pixel supplied or local WCS estimate")
    export_header["INPUTUNI"] = unit
    export_header["BAND"] = band
    export_header["HISTORY"] = "Structural filter score; not calibrated photometry or confirmed spiral arms"
    export_header["HISTORY"] = "No absolute distance or fluid microphysics inferred"
    products = []
    for name, item in maps.items():
        score_header = export_header.copy()
        score_header["BUNIT"] = "structure score"
        score_header["FILTERID"] = name
        filename = f"{name}.fits"
        fits.writeto(out / filename, item.score.astype(np.float32), score_header, overwrite=False)
        products.append(filename)
    mask_header = export_header.copy()
    mask_header["BUNIT"] = "boolean valid"
    mask_header["HISTORY"] = "1=common valid analysis annulus; 0=excluded"
    fits.writeto(out / "common_valid_mask.fits", common.astype(np.uint8), mask_header, overwrite=False)
    products.append("common_valid_mask.fits")
    for method, filename in (("linear", "grayscale.png"), ("asinh", "asinh.png"),
                             ("clahe", "clahe.png"), ("binary", "binary.png")):
        view = display_transform(np.where(image.valid, data, np.nan), method)
        # imsave writes the image at its native rectangular pixel dimensions.
        plt.imsave(out / filename, view, origin="lower", cmap="gray", vmin=0, vmax=1)
        products.append(filename)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for ax, (name, item) in zip(axes.flat, maps.items()):
        values = item.score[item.valid]
        lo, hi = np.percentile(values, [5, 99])
        if hi <= lo:
            hi = lo + 1
        ax.imshow(item.score, origin="lower", cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.image.name} / {band}: structural filters, not identified arms")
    fig.savefig(out / "filter_gallery.png", dpi=140); plt.close(fig)
    products.append("filter_gallery.png")
    result = {
        "command": "filter", "input_path": str(args.image.resolve()), "input_hdu": args.hdu,
        "inverse_variance_hdu": args.invvar_hdu, "input_shape": list(data.shape),
        "input_unit": unit, "input_band": band, "input_calibration_verified": False,
        "unit_status": "BUNIT supplied by input; calibration not independently verified" if unit != "unspecified" else "BUNIT missing: input units and absolute calibration unknown",
        "input_copied_or_modified": False,
        "pixel_scale_arcsec": pixel_scale, "pixel_scale_source": "explicit override" if args.pixel_scale_arcsec is not None else "geometric mean of celestial WCS plane pixel scales",
        "wcs_pixel_scales_arcsec": scales.tolist() if scales is not None else None,
        "geometry": {"center_xy": list(center), "pa_image_deg": args.pa_deg,
            "inclination_deg": args.inclination_deg, "inner_arcsec": args.inner_arcsec,
            "outer_arcsec": args.outer_arcsec, "source": geometry_source},
        "scales_sigma_arcsec": list(args.scales_arcsec),
        "filters": {name: item.metadata for name, item in maps.items()},
        "display_transformations": {"grayscale.png": "linear clipped percentile 1,99.5 display",
            "asinh.png": "asinh(10*x)/asinh(10) of clipped display",
            "clahe.png": "CLAHE, clip_limit=0.02, display only",
            "binary.png": "threshold 0.5 of linear clipped display; not confirmed arms"},
        "products": products,
        "limitations": ["Structural scores and display contrast are not calibrated photometric measurements.",
            "No automatic arm identification, PSF estimation, source classification, distance or viscosity inference is performed.",
            "Default zero inclination is a face-on assumption; image-center/default PA are not fitted geometry.",
            "Pixel scale is treated as isotropic; supplied geometry and WCS can be uncertain.",
            "Invalid/zero inverse-variance pixels are excluded; a four-largest-sigma guard can remove substantial annulus area.",
            "The original FITS is read in place and is not modified or duplicated."]}
    (out / "metadata.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    (out / "README.txt").write_text(
        "Galaxy structural filters\n\n"
        "Inspect metadata.json for input units, external geometry, angular scales and valid area.\n"
        "common_valid_mask.fits uses 1=valid, 0=excluded. All score FITS use this same annulus.\n"
        "FITS products preserve WCS when available. Their values are structural scores, not flux estimates.\n"
        "grayscale/asinh/CLAHE/binary PNGs are display transformations; a binary patch is not an identified arm.\n"
        "The input FITS was read in place and remains unchanged.\n"
        "No distance, fluid parameters or viscosity is inferred from these products.\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Angular galaxy measurements and conservative fluid-halo inspection")
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure", help="measure a FITS image; never infer absolute distance")
    measure.add_argument("image", type=Path)
    measure.add_argument("--out", type=Path)
    measure.add_argument("--ra", type=float)
    measure.add_argument("--dec", type=float)
    measure.add_argument("--pa-sky", type=float)
    measure.add_argument("--inclination", type=float)
    measure.add_argument("--r-inner", type=float, help="deprojected inner annulus radius in pixels")
    measure.add_argument("--r-outer", type=float, help="deprojected outer annulus radius in pixels")
    measure.add_argument("--distance-mpc", type=float, help="externally supplied angular-diameter distance")
    pair = commands.add_parser("pair", help="angular pair separation, optionally conditional projected kpc")
    for name in ("ra1", "dec1", "ra2", "dec2"):
        pair.add_argument("--" + name, type=float, required=True)
    pair.add_argument("--distance-mpc", type=float)
    filtering = commands.add_parser("filter", help="export angular-scale score FITS and separate grayscale/contrast previews")
    filtering.add_argument("image", type=Path)
    filtering.add_argument("--hdu", type=_hdu_selector, default=0, help="2D science HDU index or name")
    filtering.add_argument("--invvar-hdu", type=_hdu_selector, help="optional matching 2D inverse-variance HDU")
    filtering.add_argument("--output-dir", type=Path, required=True, help="new or empty output directory")
    filtering.add_argument("--pixel-scale-arcsec", type=float, help="positive scale; default from celestial WCS")
    filtering.add_argument("--center-x", type=float)
    filtering.add_argument("--center-y", type=float)
    filtering.add_argument("--pa-deg", type=float, default=0., help="image-plane major-axis PA from +x towards +y")
    filtering.add_argument("--inclination-deg", type=float, default=0., help="external disk inclination; default face-on assumption")
    filtering.add_argument("--inner-arcsec", type=float, default=3.)
    filtering.add_argument("--outer-arcsec", type=float, default=23.)
    filtering.add_argument("--scales-arcsec", type=float, nargs=3, default=[.4, .8, 1.6], help="three increasing Gaussian sigma scales in arcsec")
    filtering.add_argument("--band", help="passband label; default from FITS header")
    analysis = commands.add_parser("analyze", help="infer experimental arm regions and optional calibrated catalog redshift")
    analysis.add_argument("--image", type=Path, required=True, help="native registered grz image/invvar FITS")
    analysis.add_argument("--psf", type=Path, required=True, help="local native-sampling grz PSF stamp FITS")
    analysis.add_argument("--units", required=True, choices=["nanomaggy/pixel"], help="explicit declaration of science calibration")
    analysis.add_argument("--arm-model", type=Path, required=True, help="trusted locally trained calibrated display model")
    analysis.add_argument("--distance-arm-model", type=Path, help="matching TRAIN-only raw feature model for redshift")
    analysis.add_argument("--redshift-model", type=Path, help="trusted locally trained redshift artifact")
    analysis.add_argument("--catalog", type=Path, help="observed ugriz flux/ivar, angular radius and axis ratio CSV")
    analysis.add_argument("--galaxy-id", help="choose id in a multi-object photometry CSV")
    analysis.add_argument("--reference-marks", type=Path, help="optional GZ3D FITS used only for center and stars")
    analysis.add_argument("--foreground-mask", type=Path, help="optional native excluded-pixel mask")
    analysis.add_argument("--center-x", type=float)
    analysis.add_argument("--center-y", type=float)
    analysis.add_argument("--center-ra", type=float, help="target ICRS RA in degrees; identifies the right native tile")
    analysis.add_argument("--center-dec", type=float, help="target ICRS Dec in degrees")
    analysis.add_argument("--cosmological-conversion", action="store_true", help="explicit conditional Planck18 conversion; not validated geometric distance")
    analysis.add_argument("--output-dir", type=Path, required=True)
    guarded = commands.add_parser("analyze-guarded", parents=[analysis], add_help=False,
        help="run the frozen image model and record explicit scope/input abstention")
    guarded.add_argument("--survey", required=True, help="declared science survey; legacy-dr9 is the validated input survey")
    guarded.add_argument("--population", required=True, help="declared population; strong-resolved-spiral is in scope")
    fluid = commands.add_parser("fluid", help="inspect a hydrostatic isothermal halo, not inferred viscosity")
    fluid.add_argument("--rho0", type=float, required=True, help="central density in solar masses/kpc^3")
    fluid.add_argument("--sigma", type=float, required=True, help="1D dispersion in km/s")
    fluid.add_argument("--radii", type=float, nargs="+", default=[0, 1, 5, 10, 20])
    kinematics = commands.add_parser("kinematics", help="diagnose independent MaNGA Halpha velocities in angular rings")
    kinematics.add_argument("--maps", type=Path, required=True)
    kinematics.add_argument("--center-ra", type=float, required=True)
    kinematics.add_argument("--center-dec", type=float, required=True)
    kinematics.add_argument("--axis-ratio", type=float, required=True)
    kinematics.add_argument("--pa-deg", type=float, required=True, help="external photometric PA east of north")
    kinematics.add_argument("--half-light-radius-arcsec", type=float, required=True)
    kinematics.add_argument("--psf-fwhm-arcsec", type=float, required=True)
    kinematics.add_argument("--quality-flags", type=int, default=0, help="additional observation-quality flags")
    kinematics.add_argument("--output-dir", type=Path, required=True)
    btfr = commands.add_parser("btfr-distance", help="conditional distance from a matching independently measured HI observable")
    btfr.add_argument("--model", type=Path, required=True, help="local numerical posterior NPZ")
    btfr.add_argument("--velocity-kms", type=float, required=True)
    btfr.add_argument("--velocity-error-kms", type=float, required=True)
    btfr.add_argument("--velocity-definition", required=True, choices=["HI_Vflat_kms","HI_Wm50_over2_kms"])
    btfr.add_argument("--baryon-flux-proxy", type=float, required=True, help="coherent flux-derived mass proxy in Msun/Mpc^2")
    btfr.add_argument("--log-flux-error-dex", type=float, required=True)
    btfr.add_argument("--zero-point-sigma-mag", type=float, default=.05, help="assumed global distance-scale uncertainty")
    btfr.add_argument("--out", type=Path)
    screen = commands.add_parser("screen", help="record acceptance/abstention for an experimental image-redshift analysis")
    screen.add_argument("--analysis-dir", type=Path, required=True)
    screen.add_argument("--survey", required=True, help="declared science survey; only legacy-dr9 is in scope")
    screen.add_argument("--population", required=True, help="declared population; only strong-resolved-spiral is in scope")
    args = parser.parse_args(argv)
    if args.command == "measure":
        from .morphology import load_fits, measure_image, sky_center_and_pa
        loaded = load_fits(args.image)
        params = dict(pixel_scale_arcsec=loaded.metadata.get("pixel_scale_arcsec"),
                      distance_mpc=args.distance_mpc)
        if (args.ra is None) != (args.dec is None):
            parser.error("--ra and --dec must be supplied together")
        if args.pa_sky is not None and args.ra is None:
            parser.error("--pa-sky requires --ra and --dec")
        if args.ra is not None:
            center, pa = sky_center_and_pa(loaded.metadata, args.ra, args.dec, args.pa_sky)
            params["center_xy"] = center
            if pa is not None:
                params["pa_deg"] = pa
        for key, value in [("inclination_deg", args.inclination),
                           ("r_inner_px", args.r_inner), ("r_outer_px", args.r_outer)]:
            if value is not None:
                params[key] = value
        result = measure_image(loaded.data, **params)
        result["fits_metadata"] = loaded.metadata
    elif args.command == "filter":
        try:
            result = _filter_command(args)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            parser.error(str(exc))
    elif args.command in ("analyze", "analyze-guarded"):
        from .workbench import analyze
        try:
            result = analyze(args)
            if args.command == "analyze-guarded":
                from .release_guard import write_screen
                result["release_decision"] = write_screen(args.output_dir,survey=args.survey,population=args.population)
                result["report"] = str((args.output_dir/"release.html").resolve())
                result["status"] = result["release_decision"]["status"]
        except (ValueError, KeyError, IndexError, OSError) as exc:
            parser.error(str(exc))
    elif args.command == "kinematics":
        from .kinematic_workbench import analyze_maps
        try:
            result = analyze_maps(args.maps,args.output_dir,center_ra_deg=args.center_ra,
                center_dec_deg=args.center_dec,axis_ratio=args.axis_ratio,pa_deg=args.pa_deg,
                half_light_radius_arcsec=args.half_light_radius_arcsec,
                psf_fwhm_arcsec=args.psf_fwhm_arcsec,observation_quality_flags=args.quality_flags)
        except (ValueError,KeyError,OSError) as exc:
            parser.error(str(exc))
    elif args.command == "btfr-distance":
        from .erroraware_btfr import ErrorAwareBTFR
        try:
            if args.velocity_kms<=0:raise ValueError("Velocity must be positive")
            model = ErrorAwareBTFR.load(args.model)
            diagnostic = model.predict(args.velocity_kms,args.baryon_flux_proxy,
                args.velocity_error_kms/args.velocity_kms/np.log(10),args.log_flux_error_dex,
                velocity_definition=args.velocity_definition,zero_point_sigma_mag=args.zero_point_sigma_mag)
        except (ValueError,KeyError,OSError) as exc:
            parser.error(str(exc))
        accepted = not diagnostic["outside_velocity_calibration_range"]
        result = dict(status="conditional_kinematic_distance" if accepted else "abstain_velocity_extrapolation",
            accepted_distance_mpc=diagnostic["distance_mpc"] if accepted else None,
            diagnostic=diagnostic,calibration=model.summarize())
    elif args.command == "screen":
        from .release_guard import write_screen
        result = write_screen(args.analysis_dir,survey=args.survey,population=args.population)
    elif args.command == "pair":
        from .morphology import projected_pair_separation
        result = projected_pair_separation(args.ra1, args.dec1, args.ra2, args.dec2,
                                           distance_mpc=args.distance_mpc)
    else:
        from .fluid import IsothermalFluidHalo
        halo = IsothermalFluidHalo(args.rho0, args.sigma)
        radius = np.asarray(args.radii)
        result = dict(r_kpc=radius, density_msun_kpc3=halo.density(radius),
                      enclosed_mass_msun=halo.enclosed_mass(radius),
                      circular_speed_kms=halo.circular_velocity(radius),
                      scale_radius_kpc=halo.scale_radius_kpc,
                      interpretation="Self-gravitating nonrotating isothermal halo; no viscosity is inferred; outer truncation needed")
    encoded = json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
    if getattr(args, "out", None):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
