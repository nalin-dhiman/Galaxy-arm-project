#!/usr/bin/env python3
"""Freeze a fresh independent Galaxy Builder trace cohort, then cache public data."""
from pathlib import Path
import hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'executed_next/geometry'
SOURCES=OUT/'sources'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def selection():
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    catalog=pd.read_csv(SOURCES/'galaxy-builder-catalog.csv',dtype={'dr7objid':str,'galaxy_builder_id':str})
    subjects=pd.read_csv(SOURCES/'hierarchical-modelling-of-spiral-pitch-angle/lib/galaxy-builder-subjects.csv',dtype=str)
    # Repeated memberships in subject sets must refer to identical public frames.
    if (subjects.groupby('subject_id')['locations'].nunique()>1).any():
        raise ValueError('Conflicting subject frame locations; selection needs a source audit')
    subjects=subjects.drop_duplicates('subject_id',keep='first').set_index('subject_id')
    tree=json.loads((SOURCES/'hierarchical-modelling-of-spiral-pitch-angle_tree.json').read_text())
    available={r['path'] for r in tree['tree'] if r['type']=='blob'}
    previous=[]
    previous_paths=['executed_filters/annotations/selected_targets.json','executed_solid/data/selected_targets.json','executed_next/external/data/selected_targets.json']
    for relative in previous_paths:
        p=ROOT/relative
        if p.exists():previous.extend(json.loads(p.read_text())['targets'])
    coords=SkyCoord([r['ra_deg'] for r in previous]*u.deg,[r['dec_deg'] for r in previous]*u.deg)
    records=[]
    for _,r in catalog.iterrows():
        sid=r.galaxy_builder_id
        if r.is_validation or not(r.n_spirals>=1 and r.disk_q>=.4):continue
        if sid in ['20901989','20901990'] or sid not in subjects.index:continue
        if f'lib/aggregation_results/{sid}.pkl.gz' not in available:continue
        nearest=float(SkyCoord(float(r.ra)*u.deg,float(r.dec)*u.deg).separation(coords).min().arcsec)
        if nearest<10:continue
        locations=json.loads(subjects.loc[sid,'locations'])
        if '0' not in locations or not locations['0'].endswith('.json'):continue
        key=hashlib.sha256(('geometry-v1:'+sid).encode()).hexdigest()
        records.append({'id':sid,'dr7objid':r.dr7objid,'ra_deg':float(r.ra),'dec_deg':float(r.dec),
            'catalog_n_spirals':int(r.n_spirals),'catalog_axis_ratio':float(r.disk_q),
            'nearest_prior_or_parallel_sample_arcsec':nearest,'selection_hash':key,
            'aggregate_url':f'https://raw.githubusercontent.com/tingard/hierarchical-modelling-of-spiral-pitch-angle/master/lib/aggregation_results/{sid}.pkl.gz',
            'image_url':locations['0']})
    records.sort(key=lambda x:x['selection_hash'])
    if len(records)<24:raise ValueError('Fewer than24 eligible independent trace galaxies')
    selected=records[:24]
    for i,row in enumerate(selected):
        row['rank']=i;row['split']='development' if i%2==0 else 'test'
        row['aggregate_path']=str((OUT/'references'/f"{row['id']}_aggregate.pkl.gz").relative_to(ROOT))
        row['image_path']=str((OUT/'references'/f"{row['id']}_image.json").relative_to(ROOT))
    result={'source':'Galaxy Builder published human centerline polylines, distinct from GZ3D polygons',
        'candidate_catalog_rows':len(catalog),'eligible_after_fixed_cuts':len(records),
        'selection':'nonvalidation, >=1 published arm, catalogq>=0.4; exclude two schema pilots; require public reference/image records; minimum10arcsec from all old32+160 and parallel99; ascendingSHA256 geometry-v1:id',
        'selection_uses_automatic_geometry_scores':False,'targets':selected,
        'catalog_sha256':sha(SOURCES/'galaxy-builder-catalog.csv'),
        'previous_and_parallel_selection_hashes':{p:sha(ROOT/p) for p in previous_paths if (ROOT/p).exists()}}
    OUT.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    return result

def fetch(row,key):
    path=ROOT/row[key+'_path'];url=row[key+'_url']
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        last=None
        for attempt in range(3):
            try:
                response=requests.get(url,timeout=(20,120));response.raise_for_status();path.write_bytes(response.content);break
            except Exception as exc:
                last=exc
                if attempt<2:time.sleep(2*(attempt+1))
        else:raise last
    return dict(id=row['id'],kind=key,path=str(path.relative_to(ROOT)),url=url,sha256=sha(path),bytes=path.stat().st_size)

def main():
    data=selection();rows=[];failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(fetch,r,key):(r,key) for r in data['targets'] for key in ['aggregate','image']}
        for task in as_completed(tasks):
            row,key=tasks[task]
            try:rows.append(task.result());print('cached',row['id'],key,flush=True)
            except Exception as exc:failures.append({'id':row['id'],'kind':key,'reason':str(exc)})
    result={'selection_sha256':sha(OUT/'selection.json'),'files':rows,'failures':failures}
    (OUT/'source_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FILES',len(rows),'FAILURES',len(failures),flush=True)
if __name__=='__main__':main()
