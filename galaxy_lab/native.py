"""Read native, already registered Legacy-style grz image/invvar products.

No science reprojection, invented units or missing-band synthesis. The caller
must declare calibration when FITS BUNIT is absent. PSFs are local finite-stamp
second-moment Gaussian equivalents, not exact non-Gaussian core FWHMs.
"""
from pathlib import Path
import hashlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import map_coordinates
from .filtering import ScienceImage


def psf_moment_fwhm(stamp, pixel_scale_arcsec):
    values = np.maximum(np.asarray(stamp, float), 0.)
    if values.ndim != 2 or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("PSF stamp must have finite positive integrated weight")
    yy, xx = np.indices(values.shape)
    total = values.sum(); x = (xx * values).sum() / total; y = (yy * values).sum() / total
    variance = (((xx-x)**2 + (yy-y)**2) * values).sum() / total / 2
    if variance <= 0:
        raise ValueError("PSF is unresolved or singular")
    return float(2.354820045 * np.sqrt(variance) * pixel_scale_arcsec)


def select_native_grid(hdus, target_sky=None):
    """Select a common native tile, never silently take a neighbouring last HDU.

    Legacy subimages can append several bricks. A complete grid must have all
    g/r/z image+variance planes with identical celestial WCS and shape. When a
    target sky coordinate is supplied, its containing footprint is mandatory.
    Largest area then earliest HDU resolves ties without consulting image flux.
    """
    grids = {}
    for index, hdu in enumerate(hdus):
        band, kind = str(hdu.header.get("BAND", "")).strip(), str(hdu.header.get("IMAGETYP", "")).strip()
        if band not in ("g", "r", "z") or kind not in ("image", "invvar") or hdu.data is None or hdu.data.ndim != 2:
            continue
        wcs = WCS(hdu.header).celestial
        if not wcs.has_celestial:
            continue
        key = (str(hdu.header.get("BRICK", "")), tuple(hdu.data.shape), wcs.to_header().tostring())
        grid = grids.setdefault(key, {"planes": {}, "wcs": wcs, "shape": hdu.data.shape,
                                     "brick": key[0], "first_hdu": index})
        if (band, kind) in grid["planes"]:
            previous = hdus[grid["planes"][band, kind]]
            if (not np.array_equal(previous.data, hdu.data, equal_nan=True)
                    or str(previous.header.get("BUNIT", "")) != str(hdu.header.get("BUNIT", ""))):
                raise ValueError("Conflicting duplicate native planes on the same grid")
            # Redundant copies are scientifically identical. Resolve to the
            # audited catalogue's later-copy convention for literal HDU parity.
        grid["planes"][band, kind] = index
    candidates = []
    for grid in grids.values():
        if not all((b, kind) in grid["planes"] for b in "grz" for kind in ("image", "invvar")):
            continue
        if target_sky is not None:
            x, y = grid["wcs"].world_to_pixel(target_sky)
            if not np.isfinite([x, y]).all() or not (-.5 <= x < grid["shape"][1]-.5 and -.5 <= y < grid["shape"][0]-.5):
                continue
        candidates.append(grid)
    if not candidates:
        raise ValueError("No common target-containing native image/Inverse-variance WCS grid for all grz bands")
    if target_sky is None and len(candidates) > 1:
        raise ValueError("Multiple native tiles: supply a target sky coordinate to identify the galaxy footprint")
    chosen = sorted(candidates, key=lambda g: (-int(np.prod(g["shape"])), g["first_hdu"]))[0]
    return chosen, len(grids)


def read_native_grz(image_path, psf_path, *, declared_unit, mask_path=None, target_sky=None):
    """Return native grz ScienceImages and a provenance/quality dictionary.

    Input format: per-band BAND=g/r/z, IMAGETYP=image/invvar HDUs. PSF HDUs
    contain BAND or occur in g/r/z order, sampled on the native image grid.
    A user mask has 1=excluded and must share exactly this native grid.
    """
    if declared_unit != "nanomaggy/pixel":
        raise ValueError("This inference schema requires declared nanomaggy/pixel science calibration")
    with fits.open(image_path, memmap=False) as hdus:
        chosen, grid_count = select_native_grid(hdus, target_sky)
        indices = chosen["planes"]
        planes = {key: (np.array(hdus[index].data, float), hdus[index].header.copy()) for key, index in indices.items()}
    with fits.open(psf_path, memmap=False) as hdus:
        stamps = {}
        for hdu in hdus:
            if hdu.data is not None and hdu.data.ndim == 2:
                band = str(hdu.header.get("BAND", "")).strip() or "grz"[len(stamps)]
                if band in ("g", "r", "z"):
                    stamps[band] = np.array(hdu.data, float)
    mask = None
    mask_wcs = None
    if mask_path is not None:
        mask_data, mask_header = fits.getdata(mask_path, header=True)
        mask = np.asarray(mask_data, bool)
        candidate = WCS(mask_header).celestial
        mask_wcs = candidate if candidate.has_celestial else None
    images, quality = {}, {}
    for band in "grz":
        if (band, "image") not in planes or (band, "invvar") not in planes or band not in stamps:
            raise ValueError("Every g/r/z band requires image, inverse variance and measured local PSF")
        data, header = planes[band, "image"]; invvar, invheader = planes[band, "invvar"]
        bunit = str(header.get("BUNIT", "")).lower().replace(" ", "")
        if bunit and bunit not in ("nanomaggy/pixel", "nanomaggies/pixel", "nanomaggy/pix", "nanomaggy", "nanomaggies"):
            raise ValueError("Input BUNIT contradicts declared nanomaggy/pixel calibration")
        wcs, ivwcs = WCS(header).celestial, WCS(invheader).celestial
        if not wcs.has_celestial or not ivwcs.has_celestial or invvar.shape != data.shape:
            raise ValueError("Image and inverse variance require matching two-dimensional celestial grids")
        scales = proj_plane_pixel_scales(wcs) * 3600
        if not np.isfinite(scales).all() or np.min(scales) <= 0 or max(scales) / min(scales) > 1.01:
            raise ValueError("Native pixels must have near-isotropic angular sampling")
        scale = float(np.mean(scales))
        yy, xx = np.meshgrid(np.linspace(0, data.shape[0]-1, 3), np.linspace(0, data.shape[1]-1, 3), indexing="ij")
        offsets = wcs.pixel_to_world(xx, yy).separation(ivwcs.pixel_to_world(xx, yy)).arcsec
        if not np.isfinite(offsets).all() or np.max(offsets) > .01 * scale:
            raise ValueError("Inverse-variance WCS differs from the science grid")
        valid = np.isfinite(data) & np.isfinite(invvar) & (invvar > 0)
        native_noise = float(np.median(1. / np.sqrt(invvar[valid]))) if valid.any() else np.nan
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError("Foreground mask must share the native shape")
            if mask_wcs is not None:
                offsets = wcs.pixel_to_world(xx, yy).separation(mask_wcs.pixel_to_world(xx, yy)).arcsec
                if not np.isfinite(offsets).all() or np.max(offsets) > .01 * scale:
                    raise ValueError("Foreground mask WCS differs from the native grid")
            valid &= ~mask
        if not valid.any():
            raise ValueError("No finite positive-variance native science pixels")
        variance = np.divide(1., invvar, out=np.full(data.shape, np.nan), where=valid)
        fwhm = psf_moment_fwhm(stamps[band], scale)
        images[band] = ScienceImage(data, scale, unit=declared_unit, mask=~valid,
            variance=variance, wcs=wcs, psf_fwhm_arcsec=fwhm, band=band,
            pixel_value_kind="flux_per_pixel", metadata={"calibration": "caller-declared; verify source documentation",
            "psf_kind": "positive finite-stamp circular Gaussian moment equivalent", "input_bunit": header.get("BUNIT")})
        quality["psf_"+band+"_fwhm_arcsec"] = fwhm
        quality["apparent_noise_"+band+"_nanomaggy"] = native_noise
    metadata = {"image_path": str(Path(image_path).resolve()), "psf_path": str(Path(psf_path).resolve()),
        "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "psf_sha256": hashlib.sha256(Path(psf_path).read_bytes()).hexdigest(),
        "unit": declared_unit, "foreground_mask_supplied": mask_path is not None,
        "psf_assumption": "local PSF stamp sampling equals native survey pixel sampling",
        "calibration_status": "caller declared, not inferred from display contrast", "quality_features": quality}
    metadata.update(selected_native_brick=chosen["brick"], native_grids_in_file=grid_count,
        duplicate_policy="later redundant copy only after identical pixel/unit checks; conflicting copies rejected",
        selected_hdus={band: {kind+"_hdu": indices[band, kind] for kind in ("image", "invvar")} for band in "grz"},
        native_selection_rule="target-containing common grz grid; largest area then earliest HDU" if target_sky is not None else "largest complete common grz grid, earliest tie; caller must verify target centering")
    return images, metadata


def reference_center_sky(annotation_path):
    """Read only independent center marks and astrometry, without arm labels."""
    with fits.open(annotation_path) as hdus:
        centers = hdus[6].data
        if not len(centers):
            raise ValueError("No independent center marks")
        center = centers[np.argmax(centers["count"])]
        return WCS(hdus[1].header).celestial.pixel_to_world(float(center["x"]), float(center["y"]))


def annotation_center_and_stars(annotation_path, target_image):
    """Optional independent center/star inputs; human arm labels are never read."""
    with fits.open(annotation_path) as hdus:
        source_wcs = WCS(hdus[1].header).celestial
        centers = hdus[6].data
        if not len(centers):
            raise ValueError("No independent center marks in the annotation file")
        center = centers[np.argmax(centers["count"])]
        center_world = source_wcs.pixel_to_world(float(center["x"]), float(center["y"]))
        xy = tuple(float(v) for v in target_image.wcs.world_to_pixel(center_world))
        yy, xx = np.indices(target_image.data.shape)
        xs, ys = source_wcs.world_to_pixel(target_image.wcs.pixel_to_world(xx, yy))
        stars = map_coordinates(np.asarray(hdus[2].data, float), [ys, xs], order=0,
                                mode="constant", cval=0, prefilter=False) > 0
    return xy, stars
