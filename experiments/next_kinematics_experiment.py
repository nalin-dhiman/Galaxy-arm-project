#!/usr/bin/env python3
"""Execute the fixed six-target independently observed velocity-map pilot."""
from pathlib import Path
import json
import sys
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.kinematic_workbench import analyze_maps
OUT=ROOT/'executed_next/kinematics'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='passed'
    rows=[];failures=[]
    for target in protocol['targets']:
        source=next(r for r in manifest['files'] if r['id']==target['id'])
        path=ROOT/source['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Input checksum mismatch')
        out=OUT/'maps'/target['id']
        if (out/'analysis.json').exists():
            result=json.loads((out/'analysis.json').read_text())
        else:
            try:
                result=analyze_maps(path,out,**{key:target[key] for key in ('center_ra_deg','center_dec_deg','axis_ratio','half_light_radius_arcsec','psf_fwhm_arcsec')},pa_deg=target['photometric_pa_deg'],observation_quality_flags=target['drp3qual'])
            except Exception as exc:
                failures.append(dict(id=target['id'],reason=f'{type(exc).__name__}: {exc}'));continue
        rows.append(dict(id=target['id'],plateifu=target['plateifu'],rings=len(result['fit']['rings']),
            fit_pixels=result['fit']['fit_pixels'],inclination_deg=result['fit']['inclination_deg'],
            los_residual_rms_kms=result['fit']['residual_rms_kms'],
            **result['flatness']))
        print(target['id'],rows[-1]['rings'],rows[-1]['optical_flatness_candidate'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'map_summary.csv',index=False)
    summary=dict(requested=6,completed=len(rows),failures=failures,optical_flatness_candidates=sum(r['optical_flatness_candidate'] for r in rows),
        physical_distances_reported=0,targets=rows,
        interpretation='Independent spectral maps provide angular kinematic coefficients and reveal whether optical extent/geometry are adequate. No absent HI-flat speed or IR/HI baryonic flux is fabricated. These maps are not an independent distance reference and the diagnostic error proxies are not calibrated intervals.')
    (OUT/'maps_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='targets'},indent=2))


if __name__=='__main__':main()
