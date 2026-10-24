"""Acquire a new, spatially split 160-galaxy GZ3D/DR9/DR17 cohort.

Selection uses existing human classification metadata only, excludes all
previously viewed 32 galaxies, and is frozen before image/label inspection.
Network is opt-in (--download); the default inventories cached inputs only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import threading
import requests
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.acquire_arm_annotations import inventory_target, download
from experiments.acquire_filter_images import inventory as inventory_native

OUT = ROOT/'executed_solid/data'
OLD = ROOT/'executed_filters/annotations/selected_targets.json'
METADATA = ROOT/'executed_filters/annotations/gz3d_metadata.fits'
BASE = 'https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/'
DRPURL = 'https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits'
DRP = OUT/'drpall-v3_1_1.fits'
_THREAD_STATE=threading.local()
_RATE_LOCK=threading.Lock()
_LEGACY_NEXT=0.0
_LEGACY_COOLDOWN=0.0
_RATE_LIMIT_COUNT=0

def session_get(url, **kwargs):
    global _LEGACY_NEXT,_LEGACY_COOLDOWN,_RATE_LIMIT_COUNT
    is_legacy='www.legacysurvey.org/' in url
    if is_legacy:
        while True:
            with _RATE_LOCK:
                now=time.monotonic()
                delay=max(_LEGACY_NEXT,_LEGACY_COOLDOWN)-now
                if delay<=0:
                    _LEGACY_NEXT=now+2.0
                    break
            time.sleep(min(delay,1.0))
    session=getattr(_THREAD_STATE,'session',None)
    if session is None:
        session=requests.Session()
        _THREAD_STATE.session=session
    response=session.get(url,**kwargs)
    if is_legacy and response.status_code==429:
        raw=response.headers.get('Retry-After')
        try:
            retry_after=float(raw)
        except (ValueError,TypeError):
            try:
                retry_after=max(0.,parsedate_to_datetime(raw).timestamp()-time.time())
            except (ValueError,TypeError,AttributeError):
                retry_after=0.
        with _RATE_LOCK:
            _RATE_LIMIT_COUNT+=1
            cooldown=max(retry_after,60. if _RATE_LIMIT_COUNT==1 else 120.)
            _LEGACY_COOLDOWN=max(_LEGACY_COOLDOWN,time.monotonic()+cooldown)
            event=dict(at_utc=datetime.now(timezone.utc).isoformat(),status=429,url=url,
                       retry_after_header=raw,shared_cooldown_seconds=cooldown,
                       policy='At most two Legacy requests in flight; global two-second spacing; honor Retry-After and shared 60/120sec backoff')
            with (OUT/'rate_limit_events.jsonl').open('a') as stream:
                stream.write(json.dumps(event)+'\n')
            print(json.dumps(dict(rate_limited=True,shared_cooldown_seconds=cooldown)),flush=True)
    return response


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def selection():
    prior_ids = {r['id'] for r in json.loads(OLD.read_text())['targets']}
    table = fits.getdata(METADATA, 1)
    fraction = table['GZ_spiral_votes']/np.maximum(table['GZ_total_classifications'], 1)
    eligible = ((table['observed']==1) & (table['GZ_total_classifications']>=20)
                & (fraction>=.7) & (table['number_galaxy_centers']==1))
    candidates = [i for i in np.flatnonzero(eligible) if table[i]['MANGAID'].strip() not in prior_ids]
    candidates.sort(key=lambda i: digest('solid-v1:'+table[i]['MANGAID'].strip()))
    candidates = candidates[:160]
    # Sky tile groups are computed without redshift, flux, morphology scores,
    # arm mask contents, distance or any model outcomes.
    groups = {f"ra{int(float(table[i]['ra'])//10):02d}_dec{int((float(table[i]['dec'])+90)//10):02d}" for i in candidates}
    ordered_groups = sorted(groups, key=lambda group: digest('solid-v1:split:'+group))
    ntrain = int(round(.6*len(groups)))
    ncal = int(round(.2*len(groups)))
    assignment = {group: ('train' if rank<ntrain else 'calibration' if rank<ntrain+ncal else 'test')
                  for rank, group in enumerate(ordered_groups)}
    records = []
    for rank, index in enumerate(candidates):
        row = table[index]
        mid = row['MANGAID'].strip()
        group = f"ra{int(float(row['ra'])//10):02d}_dec{int((float(row['dec'])+90)//10):02d}"
        filename = 'gz3d_'+row['file_name'].strip()+'.gz'
        records.append(dict(id=mid, MANGAID=mid, rank=rank, sample_group='strong_spiral',
                            split=assignment[group], sky_group=group, group_id=group,
                            ra_deg=float(row['ra']), dec_deg=float(row['dec']),
                            RA=float(row['ra']), DEC=float(row['dec']), nsa_id=int(row['nsa_id']),
                            gz_total_classifications=int(row['GZ_total_classifications']),
                            gz_spiral_votes=int(row['GZ_spiral_votes']),
                            gz_spiral_vote_fraction=float(fraction[index]),
                            source_url=BASE+filename,
                            path=str((OUT/'annotations'/filename).relative_to(ROOT)),
                            rgb_hdu=0, center_hdu=1, stars_hdu=2, spiral_hdu=3, bar_hdu=4,
                            metadata_hdu=5, center_clusters_hdu=6, raw_spiral_classifications_hdu=9,
                            wcs_hdu=1, pixel_scale_arcsec=.099,
                            source='SDSS DR17 Galaxy Zoo 3D v4_0_0',
                            image_kind='Rendered gri display image; not calibrated individual-band flux'))
    manifest = dict(namespace='solid-v1', selected_at_policy='Before image acquisition or algorithm/label-score inspection',
                    selection=dict(observed=True, gz_total_classifications_minimum=20,
                                   spiral_vote_fraction_minimum=.7, number_galaxy_centers=1,
                                   order='sha256(solid-v1:MANGAID)', requested_size=160,
                                   previous_ids_excluded=sorted(prior_ids),
                                   eligible_before_exclusion=int(eligible.sum()), eligible_after_exclusion=int(eligible.sum())-sum(
                                       table[i]['MANGAID'].strip() in prior_ids for i in np.flatnonzero(eligible)),
                                   failed_download_policy='retain failures; no replacements',
                                   score_or_mask_selection=False),
                    split_policy=dict(group='10deg RA x10deg Dec tiles, Dec indexed from -90deg',
                                      order='sha256(solid-v1:split:sky_group)',
                                      requested_group_fractions=dict(train=.6, calibration=.2, test=.2),
                                      groups=len(groups), group_assignment=assignment,
                                      limitations='Hard tile borders do not guarantee separation of immediately adjacent objects across borders'),
                    metadata_path=str(METADATA.relative_to(ROOT)), metadata_sha256=sha(METADATA),
                    previous_selection_path=str(OLD.relative_to(ROOT)), previous_selection_sha256=sha(OLD), targets=records)
    return manifest


def native_spec(target):
    record = dict(target)
    params = dict(ra=target['ra_deg'],dec=target['dec_deg'],layer='ls-dr9',pixscale=.262,size=256,bands='grz')
    record.update(annotation_path=target['path'],
                  path=str((OUT/'native_images'/f"{target['id']}_grz_native.fits").relative_to(ROOT)),
                  source_url='https://www.legacysurvey.org/viewer/fits-cutout?'+urlencode(params)+'&subimage',
                  psf_path=str((OUT/'native_images'/f"{target['id']}_psf.fits").relative_to(ROOT)),
                  psf_source_url='https://www.legacysurvey.org/viewer/coadd-psf/?'+urlencode(dict(
                      ra=target['ra_deg'],dec=target['dec_deg'],layer='ls-dr9')),
                  survey='Legacy Surveys DR9',native_coadd_pixel_scale_arcsec=.262,requested_bands=list('grz'))
    return record


def acquire_annotation(spec, offline):
    result = None
    for attempt in range(1 if offline else 3):
        result = inventory_target(spec, offline=offline)
        if result['status']=='available':
            return result
        if not offline and attempt<2:
            time.sleep(attempt+1)
    return result


def acquire_one(target, offline):
    annotation = acquire_annotation(target, offline)
    native = inventory_native(native_spec(target), offline=offline)
    return annotation, native


def manifests(selected, annotations, natives):
    common = dict(acquired_at_utc=datetime.now(timezone.utc).isoformat(),
                  selection_path=str((OUT/'selected_targets.json').relative_to(ROOT)),
                  selection_sha256=sha(OUT/'selected_targets.json'), expected_objects=len(selected['targets']),
                  acquisition_policy='All frozen targets retained; failed requests not replaced; separate annotation and Legacy queues, at most three SDSS annotation and two Legacy requests in flight, Legacy global two-second spacing and Retry-After/shared backoff',
                  acquisition_code_sha256={str(path.relative_to(ROOT)):sha(path) for path in [
                      ROOT/'experiments/acquire_solid_sample.py',ROOT/'experiments/acquire_arm_annotations.py',
                      ROOT/'experiments/acquire_filter_images.py']})
    for name, rows in [('annotations', annotations),('native_images', natives)]:
        manifest = dict(common, targets=rows, images=[r for r in rows if r.get('status')=='available'],
                        failures=[r for r in rows if r.get('status') not in ['available','pending']],
                        pending=[r['id'] for r in rows if r.get('status')=='pending'])
        if name=='annotations':
            manifest.update(source_documentation='https://www.sdss4.org/dr17/data_access/value-added-catalogs/?vac_id=galaxy-zoo-3d',
                            source_paper='https://arxiv.org/abs/2108.02065',
                            metadata=dict(metadata_path=selected['metadata_path'],metadata_sha256=selected['metadata_sha256']))
        else:
            manifest.update(source_documentation='https://www.legacysurvey.org/viewer/urls',
                            flux_documentation='https://www.legacysurvey.org/dr9/files/')
        write_json(OUT/name/'manifest.json', manifest)
    quality_catalog(selected,natives)


def choose_common_brick(hdus,ra_deg,dec_deg):
    """Geometric HDU choice; never inspect arm labels, redshifts or scores."""
    bricks={}
    for index,hdu in enumerate(hdus):
        band=str(hdu.header.get('BAND','')).strip()
        kind=str(hdu.header.get('IMAGETYP','')).strip()
        if band not in ['g','r','z'] or kind not in ['image','invvar'] or hdu.data is None:
            continue
        brick=str(hdu.header.get('BRICK','')).strip()
        bricks.setdefault(brick,{})[band,kind]=index
    candidates=[]
    for brick,indices in bricks.items():
        if any((band,kind) not in indices for band in 'grz' for kind in ['image','invvar']):
            continue
        shapes={tuple(hdus[indices[band,kind]].data.shape) for band in 'grz' for kind in ['image','invvar']}
        if len(shapes)!=1:
            continue
        inside=True;coordinates={}
        for band in 'grz':
            hdu=hdus[indices[band,'image']]
            x,y=WCS(hdu.header).celestial.world_to_pixel_values(ra_deg,dec_deg)
            height,width=hdu.data.shape
            bandinside=bool(np.isfinite(x) and np.isfinite(y) and -.5<=x<width-.5 and -.5<=y<height-.5)
            inside &= bandinside
            coordinates[band]=dict(x=float(x),y=float(y),inside=bandinside)
        shape=next(iter(shapes))
        candidates.append(dict(brick=brick,indices=indices,shape=list(shape),area=int(np.prod(shape)),
                               earliest_hdu=min(indices.values()),target_inside=inside,coordinates=coordinates))
    valid=[c for c in candidates if c['target_inside']]
    if not valid:
        raise ValueError('No common grz image/invvar brick contains the target sky coordinate')
    chosen=sorted(valid,key=lambda c:(-c['area'],c['earliest_hdu']))[0]
    return chosen,candidates


def inventory_corrected_native(spec,offline=False):
    """Keep the native pixels; explicitly choose a common target-containing brick."""
    record=inventory_native(spec,offline=offline)
    if record.get('status') not in ['available','partial_coverage']:
        return record
    try:
        with fits.open(ROOT/record['path']) as hdus:
            chosen,candidates=choose_common_brick(hdus,record['ra_deg'],record['dec_deg'])
            bands={}
            for band in 'grz':
                image_index=chosen['indices'][band,'image'];ivar_index=chosen['indices'][band,'invvar']
                image_hdu=hdus[image_index];invvar_hdu=hdus[ivar_index]
                image=image_hdu.data;ivar=invvar_hdu.data
                valid=np.isfinite(image)&np.isfinite(ivar)&(ivar>0)
                bands[band]=dict(name=band,image_hdu=image_index,invvar_hdu=ivar_index,
                    image_shape=list(image.shape),invvar_shape=list(ivar.shape),
                    image_header_bunit=image_hdu.header.get('BUNIT'),invvar_header_bunit=invvar_hdu.header.get('BUNIT'),
                    pixel_scales_arcsec=(proj_plane_pixel_scales(WCS(image_hdu.header))*3600).tolist(),
                    wcs_header={k:v for k,v in image_hdu.header.items() if k not in ['', 'COMMENT','HISTORY']},
                    brick=chosen['brick'],finite_fraction=float(np.isfinite(image).mean()),
                    requested_shape=[256,256],clipped_at_coadd_boundary=list(image.shape)!=[256,256],
                    valid_fraction=float(valid.mean()),valid_pixels=int(valid.sum()),
                    median_sigma_nanomaggy=float(np.median(1/np.sqrt(ivar[valid]))) if valid.any() else None,
                    psf_fwhm_arcsec=record.get('psf',{}).get('bands',{}).get(band,{}).get('gaussian_equivalent_fwhm_arcsec'),
                    psf_fwhm_kind='positive-stamp second-moment Gaussian equivalent, not exact core FWHM')
            record['bands']=bands;record['band_records']=list(bands.values());record['status']='available'
            record['native_hdu_selection']=dict(
                rule='Require common grz image/invvar brick whose WCS footprint contains target RA/Dec; largest pixel area, earliest HDU tie; no mosaicking or science resampling',
                selected_brick=chosen['brick'],selected_earliest_hdu=chosen['earliest_hdu'],
                selected_area_pixels=chosen['area'],target_inside=True,
                candidates=[{k:v for k,v in c.items() if k!='indices'} for c in candidates])
    except Exception as error:
        record.update(status='failed',error='Native HDU selection: '+str(error))
    return record


def quality_catalog(selected,natives):
    """Prespecified nuisance features; no outcome/label measurements involved."""
    nuisances=['psf_fwhm_'+b for b in 'grz']+['median_sigma_'+b for b in 'grz']+['flux_sigma_'+b for b in 'ugriz']
    catalog_path=OUT/'matched_catalog.csv'
    catalogs={}
    if catalog_path.exists():
        with catalog_path.open(newline='') as stream:
            catalogs={r['id']:r for r in csv.DictReader(stream)}
    native_by_id={r['id']:r for r in natives}
    rows=[]
    for target in selected['targets']:
        native=native_by_id[target['id']]
        row=dict(id=target['id'],sky_group=target['sky_group'],split=target['split'],
                 image_available=native.get('status')=='available',psf_available=native.get('psf',{}).get('status')=='available')
        for band in 'grz':
            metadata=native.get('bands',{}).get(band,{})
            row['psf_fwhm_'+band]=metadata.get('psf_fwhm_arcsec','')
            row['median_sigma_'+band]=metadata.get('median_sigma_nanomaggy','')
            row['valid_fraction_'+band]=metadata.get('valid_fraction','')
        catalog=catalogs.get(target['id'],{})
        for band in 'ugriz':
            ivar=catalog.get('flux_ivar_'+band,'')
            row['flux_sigma_'+band]=float(1/np.sqrt(float(ivar))) if ivar and float(ivar)>0 else ''
        rows.append(row)
    path=OUT/'image_quality.csv'
    temporary=path.with_suffix('.csv.part')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(path)
    write_json(OUT/'quality_feature_protocol.json',dict(
        timing='Nuisance feature whitelist fixed before new test-set model scoring',
        primary_nuisance_features=nuisances,coverage_metadata=['valid_fraction_'+b for b in 'grz'],
        matching='All frozen 160 IDs retained, missing measurements left missing; no replacements',
        purpose='Include identical PSF/noise nuisance features in PGI and PGIA comparisons, preventing depth/seeing-only gains from being attributed to arms',
        units=dict(psf_fwhm='arcsec; Gaussian positive-stamp moment equivalent',median_sigma='nanomaggy per coadd pixel',flux_sigma='nanomaggy'),
        image_quality_path=str(path.relative_to(ROOT)),image_quality_sha256=sha(path),
        excluded_model_features=['sky_group','RA','DEC','redshift_source','spec_z (except target)','absolute magnitudes','stellar masses','distance'],
        limitations=['Local PSF moment widths are approximations to full non-Gaussian PSFs.',
                     'Noise features partly proxy survey conditions; include them in both compared baselines.',
                     'Coverage metadata is retained for auditing and optional predeclared sensitivity; never choose galaxies by test performance.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select-only',action='store_true')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--catalog-only',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--original-input-manifest',type=Path,
                        help='Optional CSV of original input paths and SHA256 hashes to verify')
    args = parser.parse_args()
    if args.audit_only:
        result=audit_inputs(args.original_input_manifest)
        if result['status']!='passed':
            raise RuntimeError('Acquisition audit failed; inspect executed_solid/data/acquisition_verification.json')
        return
    for sub in ['', 'annotations','native_images']:
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    selected = selection()
    path = OUT/'selected_targets.json'
    if path.exists() and json.loads(path.read_text()) != selected:
        raise RuntimeError('Frozen selection changed; preserve existing data and investigate')
    write_json(path, selected)
    prior_annotations_path=OUT/'annotations/manifest.json'
    prior_native_path=OUT/'native_images/manifest.json'
    cached_annotations={r['id']:r for r in json.loads(prior_annotations_path.read_text()).get('targets',[])} if prior_annotations_path.exists() else {}
    cached_natives={r['id']:r for r in json.loads(prior_native_path.read_text()).get('targets',[])} if prior_native_path.exists() else {}
    annotations = [cached_annotations.get(record['id'],dict(record,status='pending')) for record in selected['targets']]
    natives = [cached_natives.get(record['id'],dict(native_spec(record),status='pending')) for record in selected['targets']]
    manifests(selected,annotations,natives)
    if args.select_only:
        counts = {split:sum(r['split']==split for r in selected['targets']) for split in ['train','calibration','test']}
        print(json.dumps(dict(selected=len(annotations),splits=counts,selection=str(path))))
        return
    catalog_match(selected, download_requested=args.download)
    if args.catalog_only:
        return
    index = {record['id']:i for i,record in enumerate(selected['targets'])}
    # Process-local injection gives the reused public-data helpers persistent
    # connections without modifying their earlier source files or outputs.
    requests.get=session_get
    # Conservative startup cooldown after the previously observed HTTP429;
    # only Legacy calls wait, so the independent SDSS queue can finish first.
    global _LEGACY_COOLDOWN
    _LEGACY_COOLDOWN=time.monotonic()+60. if args.download else 0.
    with ThreadPoolExecutor(max_workers=3) as annotation_pool, ThreadPoolExecutor(max_workers=2) as legacy_pool:
        futures={annotation_pool.submit(acquire_annotation,record,not args.download):'annotation'
                 for record in selected['targets']}
        futures.update({legacy_pool.submit(inventory_corrected_native,native_spec(record),not args.download):'native'
                        for record in selected['targets']})
        annotation_done=set();native_done=set();last_reported=-1
        for future in as_completed(futures):
            record=future.result();position=index[record['id']]
            if futures[future]=='annotation':
                annotations[position]=record;annotation_done.add(record['id'])
            else:
                natives[position]=record;native_done.add(record['id'])
            completed=len(annotation_done & native_done)
            if completed!=last_reported and (completed==1 or completed%5==0 or completed==len(annotations)):
                last_reported=completed
                manifests(selected,annotations,natives)
                print(json.dumps(dict(completed=completed,total=len(annotations),
                                      annotation_available=sum(r['status']=='available' for r in annotations),
                                      native_available=sum(r['status']=='available' for r in natives),
                                      psf_available=sum(r.get('psf',{}).get('status')=='available' for r in natives))),flush=True)
    manifests(selected,annotations,natives)
    audit_inputs(args.original_input_manifest)


def catalog_match(selected, download_requested=False):
    if not DRP.exists():
        download(DRPURL, DRP, offline=not download_requested)
    with fits.open(DRP) as hdus:
        hdus.verify('exception')
        table = hdus['MANGA'].data
        columns = {c.name.lower(): dict(format=c.format, header_unit=c.unit) for c in hdus['MANGA'].columns}
        required = ['mangaid','plateifu','nsa_z','nsa_elpetro_flux','nsa_elpetro_flux_ivar',
                    'nsa_elpetro_th50_r','nsa_elpetro_ba','nsa_sersic_ba']
        if any(name not in columns for name in required):
            raise ValueError('DRPall schema differs from expected DR17 columns')
        rows = []
        for target in selected['targets']:
            matched = table[np.asarray([str(x).strip()==target['id'] for x in table['mangaid']])]
            output = dict(id=target['id'], MANGAID=target['id'], sky_group=target['sky_group'],
                          group_id=target['group_id'], split=target['split'],
                          ra_deg=target['ra_deg'], dec_deg=target['dec_deg'],
                          catalog_matched=bool(len(matched)), catalog_match_count=len(matched))
            if len(matched):
                # Prefer a zero DRP quality mask, then stable observation ID.
                # This selects a duplicate observation, never a replacement galaxy.
                row = sorted(matched, key=lambda r: (int(r['drp3qual'])!=0, str(r['plateifu']).strip()))[0]
                z = float(row['nsa_z'])
                output.update(plateifu=str(row['plateifu']).strip(), drp3qual=int(row['drp3qual']),
                              spec_z=z if np.isfinite(z) and 0<z<1 else '',
                              redshift_valid=bool(np.isfinite(z) and 0<z<1),
                              redshift_source='DR17 DRPall NSA_Z heliocentric catalog redshift',
                              half_light_radius_arcsec=float(row['nsa_elpetro_th50_r']),
                              axis_ratio=float(row['nsa_elpetro_ba']),
                              sersic_axis_ratio=float(row['nsa_sersic_ba']))
                for band, index in zip('ugriz', range(2,7)):
                    flux = float(row['nsa_elpetro_flux'][index])
                    ivar = float(row['nsa_elpetro_flux_ivar'][index])
                    output['flux_'+band] = flux if np.isfinite(flux) else ''
                    output['flux_ivar_'+band] = ivar if np.isfinite(ivar) else ''
                    output['flux_valid_'+band] = bool(np.isfinite(flux) and np.isfinite(ivar) and ivar>0)
                output['all_ugriz_photometry_valid'] = all(output['flux_valid_'+b] for b in 'ugriz')
            else:
                output.update(spec_z='',redshift_valid=False,all_ugriz_photometry_valid=False)
            rows.append(output)
    path = OUT/'matched_catalog.csv'
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    info = dict(source_url=DRPURL,path=str(DRP.relative_to(ROOT)),sha256=sha(DRP),bytes=DRP.stat().st_size,
                matched_catalog_path=str(path.relative_to(ROOT)),matched_catalog_sha256=sha(path),
                selected=len(rows),matched=sum(row['catalog_matched'] for row in rows),
                valid_redshifts=sum(row['redshift_valid'] for row in rows),
                valid_ugriz=sum(row['all_ugriz_photometry_valid'] for row in rows),
                selection_used_redshift_or_flux=False,
                duplicate_observation_rule='Prefer DRP3QUAL==0, then lexicographic PLATEIFU; no galaxies replaced',
                redshift_label='NSA_Z heliocentric catalog redshift; not flow-corrected NSA_ZDIST or derived luminosity distance',
                observed_features=['flux_'+b for b in 'ugriz']+['half_light_radius_arcsec','axis_ratio','sersic_axis_ratio'],
                prohibited_features_excluded=['NSA_*_ABSMAG','NSA_*_MASS','NSA_ZDIST','physical radius','distance'],
                column_units=dict(flux_ugriz='nanomaggy',flux_inverse_variance='inverse nanomaggy squared',
                                  half_light_radius='arcsec',axis_ratio='dimensionless',redshift='dimensionless'),
                band_order='NSA_ELPETRO_FLUX is F,N,u,g,r,i,z; indices 2:7 retained',
                native_column_schema={name:columns[name] for name in required},
                documentation=['https://www.sdss4.org/dr17/manga/manga-data/catalogs/',
                    'https://github.com/sciserver/sqlloader/blob/master/schema/sql/MangaTables.sql'],
                limitations=['NSA fluxes are observed catalog measurements; no redshift-derived absolute quantities included.',
                             'NSA_Z is the inherited heliocentric catalog redshift, not a fresh MaNGA spectral fit performed here.',
                             'No photometric or redshift cuts altered the frozen selection; missing rows and values are retained.',
                             'Strong-spiral MaNGA selection is not representative of the full galaxy population.'])
    write_json(OUT/'catalog_manifest.json',info)
    print(json.dumps(dict(catalog_matched=info['matched'],valid_redshifts=info['valid_redshifts'],
                          valid_ugriz=info['valid_ugriz'],catalog=str(path))),flush=True)


def audit_inputs(original_input_manifest=None):
    """Audit new and preserved inputs without changing previous result files."""
    selected=json.loads((OUT/'selected_targets.json').read_text())
    annotations=json.loads((OUT/'annotations/manifest.json').read_text())['targets']
    natives=json.loads((OUT/'native_images/manifest.json').read_text())['targets']
    problems=[]
    new=[]
    def check(path, expected, category, records):
        target=Path(path)
        if not target.is_absolute():
            target=ROOT/target
        actual=sha(target) if target.exists() else None
        records.append(dict(path=str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target),
                            sha256=actual, expected_sha256=expected, category=category, valid=actual==expected))
        if actual!=expected:
            problems.append('Missing or changed '+str(target))
    for record in annotations:
        if record.get('status')=='available':
            check(record['path'],record['sha256'],'new_annotation',new)
        else:
            problems.append('Unavailable annotation '+record['id'])
    for record in natives:
        if record.get('status')=='available':
            check(record['path'],record['sha256'],'new_native',new)
        else:
            problems.append('Unavailable native image '+record['id'])
        if record.get('psf',{}).get('status')=='available':
            check(record['psf_path'],record['psf']['sha256'],'new_psf',new)
        else:
            problems.append('Unavailable PSF '+record['id'])
    cat=json.loads((OUT/'catalog_manifest.json').read_text())
    check(cat['path'],cat['sha256'],'new_catalog',new)
    check(cat['matched_catalog_path'],cat['matched_catalog_sha256'],'matched_catalog_csv',new)
    quality=json.loads((OUT/'quality_feature_protocol.json').read_text())
    check(quality['image_quality_path'],quality['image_quality_sha256'],'image_quality_csv',new)
    if selected!=selection():
        problems.append('Frozen selection differs from the deterministic predeclared policy')
    preserved=[]
    original=[]
    if original_input_manifest is not None:
        with Path(original_input_manifest).open(newline='') as stream:
            original=list(csv.DictReader(stream))
    for record in original:
        check(record['path'],record['sha256'],'original_input',preserved)
    oldann=json.loads((ROOT/'executed_filters/annotations/manifest.json').read_text())
    oldnative=json.loads((ROOT/'executed_filters/native_images/manifest.json').read_text())
    for record in oldann['targets']:
        check(record['path'],record['sha256'],'prior_annotation',preserved)
    check(oldann['metadata']['metadata_path'],oldann['metadata']['metadata_sha256'],'prior_annotation_metadata',preserved)
    for record in oldnative['targets']:
        check(record['path'],record['sha256'],'prior_native',preserved)
        check(record['psf_path'],record['psf']['sha256'],'prior_psf',preserved)
    prior_ids={r['id'] for r in json.loads(OLD.read_text())['targets']}
    ids={r['id'] for r in selected['targets']}
    if ids & prior_ids:
        problems.append('New cohort overlaps prior IDs')
    groups={}
    for r in selected['targets']:
        groups.setdefault(r['sky_group'],set()).add(r['split'])
    if any(len(partitions)!=1 for partitions in groups.values()):
        problems.append('A sky group crosses splits')
    countfits=sum(r['category'] in ['new_annotation','new_native','new_psf','new_catalog'] for r in new)
    result=dict(status='passed' if not problems else 'failed',checked_at_utc=datetime.now(timezone.utc).isoformat(),
                selected=len(ids),prior_ids_excluded=len(prior_ids),new_public_fits_hashes_checked=countfits,
                expected_new_public_fits=481,original_hashes_checked=len(original),
                original_input_preservation_checked=original_input_manifest is not None,
                prior_public_fits_hashes_checked=len(preserved)-len(original),
                sky_groups=len(groups),group_disjoint_splits=True,
                annotation_available=sum(r.get('status')=='available' for r in annotations),
                native_available=sum(r.get('status')=='available' for r in natives),
                psf_available=sum(r.get('psf',{}).get('status')=='available' for r in natives),
                catalog_matched=cat['matched'],new_inputs=new,preserved_inputs=preserved,problems=problems)
    if countfits!=481:
        result['status']='failed'
        result['problems'].append('Incomplete new FITS denominator: '+str(countfits))
    write_json(OUT/'acquisition_verification.json',result)
    print(json.dumps({key:result[key] for key in ['status','selected','new_public_fits_hashes_checked',
                                                 'original_hashes_checked','prior_public_fits_hashes_checked']}),flush=True)
    return result


if __name__=='__main__':
    main()
