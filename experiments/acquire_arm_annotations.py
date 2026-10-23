"""Acquire independently human-annotated spiral images from SDSS DR17 GZ3D.

Selection is frozen from existing human GZ votes and galaxy IDs, before any
filter measurement.  SDSS display RGB channels are not calibrated bands.
The labels are volunteer polygons, not algorithm-derived pseudo-labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'executed_filters' / 'annotations'
BASE = 'https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/'
METADATA = OUT / 'gz3d_metadata.fits'
SELECTION = dict(
    require_observed=True,
    minimum_gz_total_classifications=20,
    minimum_spiral_vote_fraction=0.7,
    number_galaxy_centers=1,
    order='ascending sha256 of stripped MANGAID',
    sample_size=32,
    strong_spiral_sample_size=24,
    lower_spiral_vote_control_size=8,
    control_maximum_gz_spiral_vote_fraction=0.1,
    control_minimum_spatial_classifications_for_confirmed_negative=10,
    split='even selected hash-rank development; odd rank test',
    algorithm_scores_used=False,
    annotation_mask_scores_used=False,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path, offline=False):
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'Not cached: {path}')
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(response.content)
        # Opening validates FITS/gzip structure, rather than trusting HTTP 200.
        with fits.open(temporary) as hdus:
            hdus.verify('exception')
        temporary.replace(path)
    return path


def select_targets():
    metadata = fits.getdata(METADATA, 1)
    fraction = metadata['GZ_spiral_votes'] / np.maximum(metadata['GZ_total_classifications'], 1)
    eligible = ((metadata['observed'] == 1)
                & (metadata['GZ_total_classifications'] >= 20)
                & (fraction >= 0.7)
                & (metadata['number_galaxy_centers'] == 1))
    candidates = sorted(np.flatnonzero(eligible), key=lambda i: hashlib.sha256(
        metadata[i]['MANGAID'].strip().encode()).hexdigest())
    control_eligible = ((metadata['observed'] == 1)
                        & (metadata['GZ_total_classifications'] >= 20)
                        & (fraction <= 0.1)
                        & (metadata['number_galaxy_centers'] == 1))
    controls = sorted(np.flatnonzero(control_eligible), key=lambda i: hashlib.sha256(
        metadata[i]['MANGAID'].strip().encode()).hexdigest())[:8]
    selected = [(i, 'strong_spiral', rank) for rank, i in enumerate(candidates[:24])]
    selected.extend((i, 'lower_spiral_vote_control', rank) for rank, i in enumerate(controls))
    targets = []
    for index, sample_group, rank in selected:
        row = metadata[index]
        mangaid = row['MANGAID'].strip()
        filename = 'gz3d_' + row['file_name'].strip() + '.gz'
        targets.append(dict(
            id=mangaid, MANGAID=mangaid, rank=rank, sample_group=sample_group,
            split='development' if rank % 2 == 0 else 'test',
            ra_deg=float(row['ra']), dec_deg=float(row['dec']),
            RA=float(row['ra']), DEC=float(row['dec']),
            nsa_id=int(row['nsa_id']),
            gz_total_classifications=int(row['GZ_total_classifications']),
            gz_spiral_votes=int(row['GZ_spiral_votes']),
            gz_spiral_vote_fraction=float(fraction[index]),
            source_url=BASE + filename,
            path=str((OUT / filename).relative_to(ROOT)),
            rgb_hdu=0, center_hdu=1, stars_hdu=2, spiral_hdu=3,
            bar_hdu=4, metadata_hdu=5, center_clusters_hdu=6,
            raw_spiral_classifications_hdu=9,
            wcs_hdu=1, pixel_scale_arcsec=0.099,
            rgb_origin='upper: original array rows share pixel coordinates with mask arrays',
            source='SDSS DR17 Galaxy Zoo 3D v4_0_0',
            image_kind='rendered SDSS RGB volunteer image; not calibrated g/r/i flux planes',
        ))
    return targets, int(eligible.sum())


def inventory_target(spec, offline=False):
    record = spec.copy()
    try:
        path = download(spec['source_url'], ROOT / spec['path'], offline)
        with fits.open(path) as hdus:
            rgb = hdus[0].data
            arm = hdus[3].data
            if rgb.shape[:2] != arm.shape or rgb.shape[-1] != 3:
                raise ValueError(f'RGB/mask shape mismatch: {rgb.shape} vs {arm.shape}')
            wcs = WCS(hdus[1].header)
            scales = proj_plane_pixel_scales(wcs) * 3600
            centers = hdus[6].data
            # Centers come from independent volunteer marks, not arm filtering.
            main_center = centers[np.argmax(centers['count'])] if len(centers) else None
            record.update(
                shape=list(rgb.shape), rgb_dtype=str(rgb.dtype),
                mask_shape=list(arm.shape), actual_pixel_scales_arcsec=scales.tolist(),
                wcs_header=dict((key, value) for key, value in hdus[1].header.items()
                                if key not in ['', 'COMMENT', 'HISTORY']),
                number_raw_spiral_classifications=len(hdus[9].data),
                number_nonblank_spiral_classifications=sum(
                    bool(json.loads(value)) for value in hdus[9].data['spiral_paths']),
                spiral_max_votes=float(np.max(arm)),
                spiral_pixels_votes_ge1=int(np.count_nonzero(arm >= 1)),
                spiral_pixels_votes_ge3=int(np.count_nonzero(arm >= 3)),
                spiral_pixels_votes_ge5=int(np.count_nonzero(arm >= 5)),
                foreground_star_pixels=int(np.count_nonzero(hdus[2].data > 0)),
                bar_pixels_votes_ge3=int(np.count_nonzero(hdus[4].data >= 3)),
                center_x=float(main_center['x']) if main_center is not None else None,
                center_y=float(main_center['y']) if main_center is not None else None,
                center_vote_count=int(main_center['count']) if main_center is not None else 0,
                sha256=sha256(path), bytes=path.stat().st_size, status='available',
            )
        record['spatial_spiral_annotation_adequate'] = record['number_raw_spiral_classifications'] >= 10
        record['confirmed_spatial_negative'] = (record['spatial_spiral_annotation_adequate']
                                                and record['spiral_pixels_votes_ge3'] == 0)
        record['spatial_label_status'] = ('adequate_spatial_annotations' if record['spatial_spiral_annotation_adequate']
                                          else 'unknown_no_adequate_spatial_annotation_participation')
        # GZ3D target RGB pixels and mask arrays already share the exact same grid.
        record['label_limitations'] = [
            'Human votes are incomplete, uncertain polygon annotations; unlabeled pixels are not guaranteed negatives',
            'Selected sample is conditional on strong existing spiral classifications, not survey representative',
            'Rendered display RGB is upsampled; 0.099 arcsec sampling is not true angular resolution',
            'No precision PSF, inverse variance, calibrated band fluxes or pixel area map in this FITS',
        ]
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    download(BASE + METADATA.name, METADATA, args.offline)
    specs, eligible = select_targets()
    selection = dict(selection=SELECTION, eligible_galaxies=eligible,
                     metadata_path=str(METADATA.relative_to(ROOT)), metadata_sha256=sha256(METADATA),
                     targets=specs)
    (OUT / 'selected_targets.json').write_text(json.dumps(selection, indent=2) + '\n')
    if args.select_only:
        print(json.dumps(dict(selected=len(specs), eligible=eligible, path=str(OUT / 'selected_targets.json'))))
        return
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda spec: inventory_target(spec, args.offline), specs))
    manifest = dict(acquired_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_documentation='https://www.sdss4.org/dr17/data_access/value-added-catalogs/?vac_id=galaxy-zoo-3d',
                    source_paper='https://arxiv.org/abs/2108.02065',
                    source_code='https://sdss-marvin.readthedocs.io/en/stable/_modules/marvin/contrib/vacs/galaxyzoo3d.html',
                    selection=SELECTION, eligible_galaxies=eligible,
                    metadata=selection,
                    targets=results, images=[r for r in results if r['status'] == 'available'],
                    failures=[r for r in results if r['status'] != 'available'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(available=len(manifest['images']), failures=len(manifest['failures']),
                         bytes=sum(r.get('bytes', 0) for r in results), path=str(OUT / 'manifest.json'))))


if __name__ == '__main__':
    main()
