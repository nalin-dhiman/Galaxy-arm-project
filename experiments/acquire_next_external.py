"""Acquire a frozen fresh spatial/morphology cohort for existing models.

This is a same-survey external-cohort test, not external-survey validation.
Every prior32+160 galaxy and every10deg sky tile touched by them is excluded.
No score, new arm mask, redshift or photometric value enters selection.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from astropy.io import fits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiments.acquire_solid_sample as backend
OUT=ROOT/'executed_next/external'
DATA=OUT/'data'
METADATA=ROOT/'executed_filters/annotations/gz3d_metadata.fits'
DRP=ROOT/'executed_solid/data/drpall-v3_1_1.fits'
BASE='https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/'
MODEL_PATHS={
    'arm_display':'executed_solid/segmentation/arm_model.joblib',
    'arm_raw_train_only':'executed_solid/segmentation/distance_feature_arm_model.joblib',
    **{name:'executed_solid/redshift/models/'+name+'/redshift_model.joblib' for name in ['P','PG','PGI','PGIA']}}
INFERENCE_PATHS=['galaxy_lab/native.py','galaxy_lab/filtering.py','galaxy_lab/segmentation.py','galaxy_lab/redshift.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def group(ra,dec):
    return f'ra{int(float(ra)//10):02d}_dec{int((float(dec)+90)//10):02d}'


def grouped_hash_sample(indices,table,label,count):
    groups={}
    for index in indices:
        row=table[index];groups.setdefault(group(row['ra'],row['dec']),[]).append(index)
    for values in groups.values():
        values.sort(key=lambda i:digest('external-v1:'+label+':'+table[i]['MANGAID'].strip()))
    ordered=sorted(groups,key=lambda value:digest('external-v1:'+label+':group:'+value))
    chosen=[]
    while len(chosen)<count and any(groups.values()):
        for value in ordered:
            if groups[value]:chosen.append(groups[value].pop(0))
            if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError('Insufficient metadata-only candidate pool for '+label)
    return chosen


def freeze():
    previous_paths=[ROOT/'executed_filters/annotations/selected_targets.json',ROOT/'executed_solid/data/selected_targets.json']
    previous=[row for path in previous_paths for row in json.loads(path.read_text())['targets']]
    old_ids={row['id'] for row in previous};old_groups={group(row['ra_deg'],row['dec_deg']) for row in previous}
    table=fits.getdata(METADATA,1)
    fraction=table['GZ_spiral_votes']/np.maximum(table['GZ_total_classifications'],1)
    bars=table['GZ_bar_votes']/np.maximum(table['GZ_total_classifications'],1)
    eligible=[i for i,row in enumerate(table) if row['observed'] and row['GZ_total_classifications']>=20
              and row['number_galaxy_centers']==1 and row['MANGAID'].strip() not in old_ids
              and group(row['ra'],row['dec']) not in old_groups]
    strata={
        'strong_spiral':[i for i in eligible if fraction[i]>=.7],
        'weak_lowbar':[i for i in eligible if .2<fraction[i]<.5 and bars[i]<=.05],
        'weak_bar_proxy':[i for i in eligible if .2<fraction[i]<.5 and bars[i]>.2],
        'low_vote_unknown':[i for i in eligible if fraction[i]<=.1 and bars[i]<=.05]}
    chosen=[(i,'strong_spiral') for i in sorted(strata['strong_spiral'],key=lambda i:digest('external-v1:strong:'+table[i]['MANGAID'].strip()))]
    for label,count in [('weak_lowbar',12),('weak_bar_proxy',12),('low_vote_unknown',16)]:
        chosen.extend((i,label) for i in grouped_hash_sample(strata[label],table,label,count))
    targets=[]
    for rank,(index,label) in enumerate(chosen):
        row=table[index];mid=row['MANGAID'].strip();filename='gz3d_'+row['file_name'].strip()+'.gz';sky_group=group(row['ra'],row['dec'])
        targets.append(dict(id=mid,MANGAID=mid,rank=rank,split='external',sample_group=label,sky_group=sky_group,group_id=sky_group,
            ra_deg=float(row['ra']),dec_deg=float(row['dec']),RA=float(row['ra']),DEC=float(row['dec']),nsa_id=int(row['nsa_id']),
            gz_total_classifications=int(row['GZ_total_classifications']),gz_spiral_votes=int(row['GZ_spiral_votes']),
            gz_spiral_vote_fraction=float(fraction[index]),gz_bar_votes=int(row['GZ_bar_votes']),gz_bar_vote_fraction=float(bars[index]),
            path=str((DATA/'annotations'/filename).relative_to(ROOT)),source_url=BASE+filename,
            rgb_hdu=0,center_hdu=1,stars_hdu=2,spiral_hdu=3,bar_hdu=4,metadata_hdu=5,center_clusters_hdu=6,raw_spiral_classifications_hdu=9,
            wcs_hdu=1,pixel_scale_arcsec=.099,source='SDSS DR17 GZ3D v4_0_0',
            image_kind='rendered SDSS gri volunteer image; not calibrated band flux'))
    selected=dict(namespace='external-v1',survey_scope='Same-survey fresh spatial/population cohort; not external-survey validation',
        selection=dict(base='observed,>=20GZclassifications,onecenter',prior_ids_excluded=sorted(old_ids),prior_sky_groups_excluded=sorted(old_groups),
            strong='ALL eligible strong spirals fraction>=.7',weak_lowbar='12group-roundrobin hash sample .2<spiralfraction<.5,barfraction<=.05',
            weak_bar_proxy='12group-roundrobin hash sample .2<spiralfraction<.5,barfraction>.2',low_vote_unknown='16group-roundrobin hash sample spiralfraction<=.1,barfraction<=.05',
            hash_namespace='external-v1',new_photometry_redshift_masks_or_scores_used=False,failed_requests_replaced=False,
            candidate_counts={key:len(value) for key,value in strata.items()}),
        metadata_path=str(METADATA.relative_to(ROOT)),metadata_sha256=sha(METADATA),previous_selection_sha256={str(path.relative_to(ROOT)):sha(path) for path in previous_paths},targets=targets)
    data_path=DATA/'selected_targets.json'
    if data_path.exists() and json.loads(data_path.read_text())!=selected:raise RuntimeError('Immutable external selection changed')
    write(data_path,selected)
    prior_features=pd.read_csv(ROOT/'executed_solid/redshift/observed_features_and_labels.csv')
    train=prior_features.loc[prior_features.split=='train']
    cuts={key:np.quantile(train[key].dropna(),[.25,.75]).tolist() for key in ['angular_half_light_radius_arcsec','axis_ratio','psf_r_fwhm_arcsec','apparent_noise_r_nanomaggy']}
    protocol=dict(frozen_models={key:dict(path=value,sha256=sha(ROOT/value)) for key,value in MODEL_PATHS.items()},
        inference_sources_sha256={path:sha(ROOT/path) for path in INFERENCE_PATHS},
        selection_sha256=sha(data_path),cohort_counts={key:sum(row['sample_group']==key for row in targets) for key in strata},
        primary_arm_endpoint='Macro galaxy AP against independent >=3vote arm-region masks in fresh strong spirals; fixed calibrated artifact threshold, >=5vote sensitivity',
        secondary_arm_endpoint='Fixed-threshold precision/recall/F1/Brier; weak-stratum protocol agreement; unasked low-vote masks remain unknown',
        primary_redshift_endpoint='Frozen PGI minus PGIA paired absolute-error difference, sky-group bootstrap, fresh strong stratum; no refit or recalibration',
        secondary_redshift='P/PG/PGI/PGIA errors, interval coverage, range/missing flags in each morphology stratum and pooled conditional cohort',
        abstention_rule='Ancillary strict-support suggested_abstention if any model predictor missing OR outside frozen training min/max. Preserve and score unchanged predictions for all eligible rows; report retained fraction and subset metrics separately.',
        image_failure_policy='Retain failed target identities and mark unavailable arm descriptors; no new galaxy replacement',
        human_arm_truth_feature_input=False,interarm_caveat='Unmarked pixels are incomplete observational annotations, not guaranteed physical negatives',
        unknown_control_caveat='Low GZ spiral votes do not establish no arms; absent spatial workflow means unknown and never a negative arm truth label',
        intervals='Keep frozen nominal90% catalog-redshift intervals; empirical coverage only, no guarantee under population/covariate shift',
        pretrained_quartile_cuts=cuts,
        gallery_ids=[row['id'] for label in strata for row in [r for r in targets if r['sample_group']==label][:2]],
        no_external_survey_claim=True)
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Frozen external model/protocol snapshot changed')
    write(protocol_path,protocol)
    return selected,protocol


def native_spec(target):
    record=dict(target);params=dict(ra=target['ra_deg'],dec=target['dec_deg'],layer='ls-dr9',pixscale=.262,size=256,bands='grz')
    record.update(annotation_path=target['path'],path=str((DATA/'native_images'/f"{target['id']}_grz_native.fits").relative_to(ROOT)),
        source_url='https://www.legacysurvey.org/viewer/fits-cutout?'+urlencode(params)+'&subimage',
        psf_path=str((DATA/'native_images'/f"{target['id']}_psf.fits").relative_to(ROOT)),
        psf_source_url='https://www.legacysurvey.org/viewer/coadd-psf/?'+urlencode(dict(ra=target['ra_deg'],dec=target['dec_deg'],layer='ls-dr9')),
        survey='Legacy Surveys DR9',native_coadd_pixel_scale_arcsec=.262,requested_bands=list('grz'))
    return record


def annotations_one(target,offline):
    result=backend.acquire_annotation(target,offline)
    if result.get('status')=='available':
        result['label_limitations']=[value if not value.startswith('Selected sample') else 'Selected morphology-stratified MaNGA sample is not population representative' for value in result['label_limitations']]
    return result


def save_manifests(selected,annotations,natives):
    common=dict(acquired_at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(DATA/'selected_targets.json'),
        expected_objects=len(selected['targets']),source_scope='same-survey fresh spatial/population external cohort',
        metadata=dict(metadata_path=selected['metadata_path'],metadata_sha256=selected['metadata_sha256']))
    for label,rows in [('annotations',annotations),('native_images',natives)]:
        write(DATA/label/'manifest.json',dict(common,targets=rows,images=[r for r in rows if r.get('status')=='available'],
            failures=[r for r in rows if r.get('status') not in ['available','pending']],pending=[r['id'] for r in rows if r.get('status')=='pending']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select-only',action='store_true');parser.add_argument('--download',action='store_true');parser.add_argument('--catalog-only',action='store_true')
    args=parser.parse_args()
    for folder in [OUT,DATA,DATA/'annotations',DATA/'native_images']:folder.mkdir(parents=True,exist_ok=True)
    selected,protocol=freeze()
    if args.select_only:
        print(json.dumps(dict(selected=len(selected['targets']),counts=protocol['cohort_counts'],sky_groups=len({r['sky_group'] for r in selected['targets']}),protocol=str(OUT/'protocol.json'))));return
    # Reuse the cached released DRPall catalog, never alter it or older outputs.
    backend.OUT=DATA;backend.DRP=DRP
    backend.catalog_match(selected,download_requested=False)
    index={r['id']:i for i,r in enumerate(selected['targets'])};annotations=[];natives=[]
    for label,destination,convert in [('annotations',annotations,lambda r:r),('native_images',natives,native_spec)]:
        path=DATA/label/'manifest.json';prior={r['id']:r for r in json.loads(path.read_text()).get('targets',[])} if path.exists() else {}
        destination.extend(prior.get(r['id'],dict(convert(r),status='pending')) for r in selected['targets'])
    save_manifests(selected,annotations,natives)
    if args.catalog_only:return
    backend.requests.get=backend.session_get;backend._LEGACY_COOLDOWN=time.monotonic()+60. if args.download else 0.
    with ThreadPoolExecutor(max_workers=3) as annotation_pool,ThreadPoolExecutor(max_workers=2) as native_pool:
        futures={annotation_pool.submit(annotations_one,r,not args.download):'annotation' for r in selected['targets']}
        futures.update({native_pool.submit(backend.inventory_corrected_native,native_spec(r),not args.download):'native' for r in selected['targets']})
        adone=set();ndone=set();last=-1
        for future in as_completed(futures):
            record=future.result();position=index[record['id']]
            if futures[future]=='annotation':annotations[position]=record;adone.add(record['id'])
            else:natives[position]=record;ndone.add(record['id'])
            complete=len(adone&ndone)
            if complete!=last and (complete==1 or complete%5==0 or complete==len(annotations)):
                last=complete;save_manifests(selected,annotations,natives)
                print(json.dumps(dict(completed=complete,total=len(annotations),annotations=sum(r.get('status')=='available' for r in annotations),
                    native=sum(r.get('status')=='available' for r in natives),psf=sum(r.get('psf',{}).get('status')=='available' for r in natives))),flush=True)
    save_manifests(selected,annotations,natives)


if __name__=='__main__':main()
