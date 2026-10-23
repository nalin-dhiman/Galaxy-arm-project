#!/usr/bin/env python3
"""Execute the predeclared global-anchor sensitivity, without model selection."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from next_btfr_experiment import table,fit,metrics,OUT,LN10


def main():
    destination=OUT/'btfr_zeropoint_sensitivity.json'
    if destination.exists():raise SystemExit('Preserve existing sensitivity')
    t=table();t=t[t.distance_method.isin([2,3])&(t.vf2019>0)&(t.inclination_deg>=30)&(t.quality<=2)].reset_index(drop=True)
    rows=[]
    for i,row in t.iterrows():
        m=fit(t.drop(i))
        for sigma in (0.,.05,.1):
            p=m.predict(row.vf2019,row.flux_proxy,row.e_vf2019/row.vf2019/LN10,row.log_flux_error,zero_point_sigma_mag=sigma)
            p.update(galaxy=row.galaxy,reference_mpc=row.distance_mpc,zero_point_sigma_mag=sigma)
            rows.append(p)
    f=pd.DataFrame(rows);f.to_csv(OUT/'btfr_zeropoint_sensitivity.csv',index=False)
    result=dict(status='completed_prespecified_sensitivity',assumption='Global anchor sigma is an external assumption, not estimated from these galaxies. Individual intervals are shown; shared calibration covariance across targets is not fitted.',
        cases={str(s):metrics(g) for s,g in f.groupby('zero_point_sigma_mag')})
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
