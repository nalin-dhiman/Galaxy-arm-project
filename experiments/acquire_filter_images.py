"""Fetch native DR9 coadd subimages, inverse variance, and local PSF stamps.

Uses the frozen independent GZ3D 32-galaxy cohort. No algorithm-score selection.
Native here means no EXTRA viewer resampling: DR9 coadds already resample the
individual exposures. Image flux and inverse variance retain per-band WCS.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import os
from urllib.parse import urlencode

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'executed_filters/native_images'
SELECTED = ROOT / 'executed_filters/annotations/selected_targets.json'
BANDS = 'grz'
PIXEL_SCALE = 0.262


def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path, offline=False):
    if path.exists():
        with fits.open(path) as hdus:
            hdus.verify('exception')
        return
    if offline:
        raise FileNotFoundError(f'Not cached: {path}')
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 120))
            response.raise_for_status()
            part = path.with_suffix(path.suffix + '.part')
            part.write_bytes(response.content)
            with fits.open(part) as hdus:
                hdus.verify('exception')
            part.replace(path)
            return
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    raise error


def psf_moments(data, pixel_scale):
    data = np.asarray(data, dtype=float)
    weight = np.clip(np.nan_to_num(data), 0, None)
    total = weight.sum()
    if total <= 0:
        raise ValueError('PSF has no positive finite weight')
    yy, xx = np.indices(data.shape)
    cx = float((weight * xx).sum() / total)
    cy = float((weight * yy).sum() / total)
    vx = float((weight * (xx - cx)**2).sum() / total)
    vy = float((weight * (yy - cy)**2).sum() / total)
    cov = float((weight * (xx - cx) * (yy - cy)).sum() / total)
    sigma_pix = np.sqrt((vx + vy) / 2)
    return dict(gaussian_equivalent_fwhm_arcsec=float(2.354820045 * sigma_pix * pixel_scale),
                sigma_pixel=float(sigma_pix), moment_variance_x=vx, moment_variance_y=vy,
                moment_covariance_xy=cov, centroid_x=cx, centroid_y=cy,
                signed_sum=float(np.nansum(data)), positive_sum=float(total),
                negative_absolute_fraction=float(np.abs(np.minimum(np.nan_to_num(data), 0)).sum()/total),
                shape=list(data.shape),
                method='Full finite 63x63 (or supplied stamp) second moment of positive PSF weights; circular trace/2; 2.35482sigma',
                interpretation='Gaussian moment-equivalent scale, not an exact PSF core FWHM; finite stamp, wings and non-Gaussian shape matter')


def specs():
    targets = json.loads(SELECTED.read_text())['targets']
    for target in targets:
        params = dict(ra=target['ra_deg'], dec=target['dec_deg'], layer='ls-dr9',
                      pixscale=PIXEL_SCALE, size=256, bands=BANDS)
        psf_params = dict(ra=target['ra_deg'], dec=target['dec_deg'], layer='ls-dr9')
        record = dict(target)
        record.update(path=str((OUT / f"{target['id']}_grz_native.fits").relative_to(ROOT)),
                      source_url='https://www.legacysurvey.org/viewer/fits-cutout?' + urlencode(params) + '&subimage',
                      psf_path=str((OUT / f"{target['id']}_psf.fits").relative_to(ROOT)),
                      psf_source_url='https://www.legacysurvey.org/viewer/coadd-psf/?' + urlencode(psf_params),
                      annotation_path=target['path'], survey='Legacy Surveys DR9',
                      native_coadd_pixel_scale_arcsec=PIXEL_SCALE,
                      requested_bands=list(BANDS))
        # The initial probe was downloaded independently by the coordinator.
        if target['id'] == '1-38856':
            record['source_url'] = 'https://www.legacysurvey.org/viewer/fits-cutout?ra=54.891785&dec=-0.51195985&layer=ls-dr9&pixscale=0.262&size=256&bands=grz&subimage'
            record['psf_source_url'] = 'https://www.legacysurvey.org/viewer/coadd-psf/?ra=54.891785&dec=-0.51195985&layer=ls-dr9'
        yield record


def inventory(spec, offline=False):
    record = dict(spec)
    path = ROOT / record['path']
    try:
        fetch(record['source_url'], path, offline)
        bands = {}
        with fits.open(path) as hdus:
            for index, hdu in enumerate(hdus):
                if hdu.data is None:
                    continue
                band = str(hdu.header.get('BAND', '')).strip()
                kind = str(hdu.header.get('IMAGETYP', '')).strip()
                if band not in BANDS or kind not in ['image', 'invvar']:
                    continue
                entry = bands.setdefault(band, {})
                entry[kind + '_hdu'] = index
                entry[kind + '_shape'] = list(hdu.data.shape)
                entry[kind + '_header_bunit'] = hdu.header.get('BUNIT')
                if kind == 'image':
                    entry.update(pixel_scales_arcsec=(proj_plane_pixel_scales(WCS(hdu.header))*3600).tolist(),
                                 wcs_header={k:v for k,v in hdu.header.items() if k not in ['', 'COMMENT', 'HISTORY']},
                                 brick=hdu.header.get('BRICK'), finite_fraction=float(np.isfinite(hdu.data).mean()))
            for band, entry in bands.items():
                if 'image_hdu' not in entry or 'invvar_hdu' not in entry:
                    raise ValueError(f'Missing image or invvar for {band}')
                image = hdus[entry['image_hdu']].data
                invvar = hdus[entry['invvar_hdu']].data
                if image.shape != invvar.shape:
                    raise ValueError('Image/invvar shape mismatch')
                valid = np.isfinite(image) & np.isfinite(invvar) & (invvar > 0)
                entry['requested_shape'] = [256, 256]
                entry['clipped_at_coadd_boundary'] = list(image.shape) != [256, 256]
                entry['valid_fraction'] = float(valid.mean())
                entry['valid_pixels'] = int(valid.sum())
                entry['median_sigma_nanomaggy'] = float(np.median(1 / np.sqrt(invvar[valid]))) if valid.any() else None
            record.update(bands=bands, bytes=path.stat().st_size, sha256=hashfile(path),
                          all_requested_bands_present=set(bands) == set(BANDS),
                          status='available' if set(bands) == set(BANDS) else 'partial_coverage')
        record['documented_flux_units'] = 'nanomaggy per coadd pixel'
        record['documented_invvar_units'] = 'inverse nanomaggy squared per coadd pixel'
        record['unit_provenance'] = 'DR9 files documentation; BUNIT absent in supplied subimage HDUs'
        record['variance_limitations'] = ['Diagonal inverse variance only; coadd resampling correlates neighboring pixels',
                                          'No survey maskbits product supplied; positive invvar is not a complete artifact mask']
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    try:
        psfpath = ROOT / record['psf_path']
        fetch(record['psf_source_url'], psfpath, offline)
        psfbands = {}
        with fits.open(psfpath) as hdus:
            for index, hdu in enumerate(hdus):
                band = str(hdu.header.get('BAND', '')).strip()
                if band in BANDS and hdu.data is not None:
                    psfbands[band] = dict(hdu=index, **psf_moments(hdu.data, PIXEL_SCALE))
        record['psf'] = dict(status='available' if set(psfbands)==set(BANDS) else 'partial_coverage',
                             bands=psfbands, bytes=psfpath.stat().st_size, sha256=hashfile(psfpath),
                             pixel_scale_arcsec=PIXEL_SCALE,
                             pixel_scale_provenance='DR9 coadd PSF stamp API on native coadd grid; .262arcsec DR9 stack sampling',
                             limitations=['Local target-center coadd PSF model, not a field-wide PSF map',
                                          'Moment-equivalent Gaussian broadening does not match complete non-Gaussian PSF wings'])
    except Exception as exc:
        record['psf'] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
    record['unit'] = 'nanomaggy_per_pixel'
    record['unit_source_url'] = 'https://www.legacysurvey.org/dr9/files/'
    for name, entry in record.get('bands', {}).items():
        entry['name'] = name
        entry['psf_fwhm_arcsec'] = record.get('psf', {}).get('bands', {}).get(name, {}).get('gaussian_equivalent_fwhm_arcsec')
        entry['psf_fwhm_kind'] = 'positive-stamp second-moment Gaussian equivalent, not exact core FWHM'
    record['band_records'] = list(record.get('bands', {}).values())
    record['psfpath'] = record['psf_path']
    return record


def manifest(results):
    return dict(acquired_at_utc=datetime.now(timezone.utc).isoformat(),
                source_documentation='https://www.legacysurvey.org/viewer/urls',
                flux_documentation='https://www.legacysurvey.org/dr9/files/',
                selection_source=str(SELECTED.relative_to(ROOT)), selection_sha256=hashfile(SELECTED),
                purpose='Same-cohort independently human-labeled arm filter comparison; not independent distance calibration',
                image_resampling='No extra viewer resampling via subimage; original coadds themselves use Lanczos resampling',
                targets=results, images=[r for r in results if r.get('status')=='available'],
                failures=[r for r in results if r.get('status')!='available'])


def alignment_preview(record):
    """Check the opposite-y WCS orientation with independent mask contours."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/galaxy_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    with fits.open(ROOT / record['path']) as native, fits.open(ROOT / record['annotation_path']) as annotated:
        image_hdu = record['bands']['r']['image_hdu']
        image = native[image_hdu].data
        yy, xx = np.indices(image.shape)
        sky = WCS(native[image_hdu].header).pixel_to_world_values(xx, yy)
        ax, ay = WCS(annotated[1].header).world_to_pixel_values(*sky)
        labels = map_coordinates(annotated[3].data, [ay, ax], order=0, mode='constant', cval=0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(annotated[0].data, origin='upper')
        axes[0].contour(annotated[3].data, levels=[3, 5], colors=['lime', 'cyan'], linewidths=.7)
        axes[0].set_title('Original SDSS volunteer RGB + human vote masks')
        lo, hi = np.percentile(image[np.isfinite(image)], [10, 99.7])
        display = np.arcsinh(np.maximum(image-lo, 0) / max((hi-lo)*.1, 1e-6))
        axes[1].imshow(display, cmap='gray', origin='lower')
        axes[1].contour(labels, levels=[3, 5], colors=['lime', 'cyan'], linewidths=.7)
        axes[1].set_title('Independent Legacy DR9 r + WCS-projected labels')
        for axis in axes:
            axis.set_axis_off()
        fig.suptitle('MANGAID '+record['id']+'; 3 votes green, 5 votes cyan; orientations differ by stored WCS')
        fig.tight_layout()
        fig.savefig(OUT/'wcs_alignment_example.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = list(specs())
    (OUT / 'targets.json').write_text(json.dumps(selected, indent=2)+'\n')
    if args.select_only:
        print(json.dumps(dict(targets=len(selected), path=str(OUT/'targets.json'))))
        return
    first = selected[0]
    if not args.offline:
        for probe, destination in [('/tmp/galaxy_native_test.fits', first['path']),
                                   ('/tmp/galaxy_psf_test.fits', first['psf_path'])]:
            if first['id']=='1-38856' and Path(probe).exists() and not (ROOT/destination).exists():
                shutil.copyfile(probe, ROOT/destination)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(inventory, spec, args.offline): spec for spec in selected}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda item: next(i for i, spec in enumerate(selected) if spec['id']==item['id']))
            temporary = OUT/'manifest.json.part'
            temporary.write_text(json.dumps(manifest(results), indent=2)+'\n')
            temporary.replace(OUT/'manifest.json')
            print(json.dumps(dict(id=result['id'], status=result['status'], psf_status=result['psf']['status'], completed=len(results))), flush=True)
    available = [r for r in results if r.get('status')=='available']
    if available:
        alignment_preview(available[0])
    print(json.dumps(dict(available=sum(r['status']=='available' for r in results),
                          psf_available=sum(r['psf']['status']=='available' for r in results), total=len(results),
                          path=str(OUT/'manifest.json'))))


if __name__ == '__main__':
    main()
