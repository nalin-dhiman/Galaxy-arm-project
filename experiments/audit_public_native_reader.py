"""Compare public inference's native reader with all frozen source planes.

Offline and read-only: no arm labels are opened, no input manifest is changed.
Both explicit sky centers and independent reference-center marks are tested.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.native import read_native_grz, reference_center_sky


def main():
    manifest_path=ROOT/'executed_solid/data/native_images/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    quality_path=ROOT/'executed_solid/data/image_quality.csv'
    with quality_path.open(newline='') as stream:
        quality={r['id']:r for r in csv.DictReader(stream)}
    results=[];max_noise=0.;max_psf=0.;start=time.monotonic()
    for target in manifest['targets']:
        record=dict(id=target['id'],split=target['split'],path=target['path'])
        try:
            sky=SkyCoord(target['ra_deg']*u.deg,target['dec_deg']*u.deg)
            images,metadata=read_native_grz(ROOT/target['path'],ROOT/target['psf_path'],
                declared_unit='nanomaggy/pixel',target_sky=sky)
            mismatches=[];bands={};all_scientific=True;all_indices=True
            with fits.open(ROOT/target['path']) as hdus:
                for band in 'grz':
                    expected=target['bands'][band];actual=metadata['selected_hdus'][band]
                    im=images[band]
                    source_image=np.asarray(hdus[expected['image_hdu']].data,dtype=float)
                    source_ivar=np.asarray(hdus[expected['invvar_hdu']].data,dtype=float)
                    valid=np.isfinite(source_image)&np.isfinite(source_ivar)&(source_ivar>0)
                    hdu_match=all(actual[k]==expected[k] for k in ['image_hdu','invvar_hdu'])
                    detail=dict(hdu_match=hdu_match,expected_hdus={k:expected[k] for k in ['image_hdu','invvar_hdu']},
                        actual_hdus=actual,shape=list(im.data.shape),shape_match=list(im.data.shape)==expected['image_shape'],
                        science_pixels_exact=im.data.shape==source_image.shape and np.array_equal(im.data,source_image,equal_nan=True),
                        valid_mask_exact=im.valid.shape==valid.shape and np.array_equal(im.valid,valid))
                    plane_checks=[]
                    for kind in ['image_hdu','invvar_hdu']:
                        a,b=hdus[expected[kind]],hdus[actual[kind]]
                        detail[kind+'_raw_bytes_exact']=a.data.tobytes()==b.data.tobytes()
                        detail[kind+'_headers_exact']=a.header.tostring()==b.header.tostring()
                        plane_checks.extend([detail[kind+'_raw_bytes_exact'],detail[kind+'_headers_exact']])
                    measured_noise=metadata['quality_features']['apparent_noise_'+band+'_nanomaggy']
                    expected_noise=float(quality[target['id']]['median_sigma_'+band])
                    measured_psf=metadata['quality_features']['psf_'+band+'_fwhm_arcsec']
                    expected_psf=float(quality[target['id']]['psf_fwhm_'+band])
                    noise_difference=abs(measured_noise-expected_noise)/max(abs(expected_noise),1e-30)
                    psf_difference=abs(measured_psf-expected_psf)/max(abs(expected_psf),1e-30)
                    max_noise=max(max_noise,noise_difference);max_psf=max(max_psf,psf_difference)
                    detail.update(noise_relative_difference=noise_difference,psf_relative_difference=psf_difference)
                    scientific=all(plane_checks) and all(detail[k] for k in ['shape_match','science_pixels_exact','valid_mask_exact'])
                    all_scientific &= scientific;all_indices &= hdu_match
                    if not all([scientific,hdu_match,noise_difference<5e-7,psf_difference<1e-12]):
                        mismatches.append(band)
                    bands[band]=detail
            center=reference_center_sky(ROOT/target['annotation_path'])
            _,reference_metadata=read_native_grz(ROOT/target['path'],ROOT/target['psf_path'],
                declared_unit='nanomaggy/pixel',target_sky=center)
            reference_match=reference_metadata['selected_hdus']==metadata['selected_hdus']
            brick_match=metadata['selected_native_brick']==target['native_hdu_selection']['selected_brick']
            if not reference_match:mismatches.append('reference_center_hdus')
            if not brick_match:mismatches.append('brick')
            record.update(status='passed' if not mismatches else 'failed',selected_brick=metadata['selected_native_brick'],
                brick_matches_manifest=brick_match,native_grids_in_file=metadata['native_grids_in_file'],
                reference_center_hdus_match=reference_match,all_selected_hdus_scientifically_identical=bool(all_scientific),
                strict_hdu_indices_match=bool(all_indices),bands=bands,mismatches=mismatches)
        except Exception as error:
            record.update(status='failed',error=f'{type(error).__name__}: {error}')
        results.append(record)
    failed=[r['id'] for r in results if r['status']!='passed']
    output=dict(status='passed' if not failed and len(results)==160 else 'failed',
        checked_at_utc=datetime.now(timezone.utc).isoformat(),targets_checked=len(results),bands_checked=3*len(results),
        passed=sum(r['status']=='passed' for r in results),mismatches=failed,
        scientifically_identical_readers=sum(r.get('all_selected_hdus_scientifically_identical',False) for r in results),
        strict_hdu_indices_match=sum(r.get('strict_hdu_indices_match',False) for r in results),
        maximum_noise_relative_difference=max_noise,maximum_psf_relative_difference=max_psf,
        noise_tolerance_relative=5e-7,psf_tolerance_relative=1e-12,
        noise_precision_note='Acquisition uses float32 IVAR arithmetic; inference uses float64. Tiny noise roundoff is expected.',
        method='Explicit target sky and independent reference-center readers compared against frozen HDU indices, exact science/IVAR bytes and headers, shape, BRICK, valid mask, PSF and noise features; no arm truth',
        elapsed_seconds=time.monotonic()-start,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        image_quality_sha256=hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        public_reader_source_sha256=hashlib.sha256((ROOT/'galaxy_lab/native.py').read_bytes()).hexdigest(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_modified=False,human_arm_truth_read=False,records=results)
    path=ROOT/'executed_solid/native_reader_audit.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='records'},indent=2))
    if output['status']!='passed':
        raise RuntimeError('Native reader integration audit failed; inspect '+str(path))


if __name__=='__main__':
    main()
