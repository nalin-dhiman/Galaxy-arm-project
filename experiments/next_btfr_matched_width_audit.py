#!/usr/bin/env python3
"""Exploratory fairness audit: identical width-available calibrators for all arms.

The original width comparison had29 vs22 training calibrators. Preserve it and
isolate the cohort-size confound here; this is not new independent test data.
"""
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from next_btfr_experiment import table,fit,predict,metrics,OUT


def main():
    destination=OUT/'btfr_matched_width_audit.json'
    if destination.exists():raise SystemExit('Preserve existing audit')
    t=table();t=t[(t.distance_method.isin([2,3]))&(t.vf2019>0)&(t.inclination_deg>=30)&(t.quality<=2)&(t.wm50>0)&(t.e_wm50>0)].copy()
    t['half_width']=t.wm50/2;t['e_half_width']=t.e_wm50/2
    rows=[]
    for i,row in t.iterrows():
        train=t.drop(i);vf=fit(train);width=fit(train,'half_width','e_half_width','HI_Wm50_over2_kms')
        reference=predict(vf,t.loc[[i]])
        right=predict(width,t.loc[[i]],'half_width','e_half_width','HI_Wm50_over2_kms')
        wrong=predict(vf,t.loc[[i]],'half_width','e_half_width','HI_Vflat_kms')
        for label,p in [('matched_Vflat',reference),('matched_width_calibration',right),('deliberate_wrong_substitution',wrong)]:
            p['comparison']=label;p['training_galaxies']=len(train);rows.append(p)
    frame=pd.concat(rows,ignore_index=True);frame.to_csv(OUT/'btfr_matched_width_predictions.csv',index=False)
    result=dict(status='completed_exploratory_fairness_audit',reason='Earlier wrong-substitution model used29 calibrators versus22 for width model; this audit fixes identical training IDs, not just target IDs.',
        target_galaxies=len(t),training_galaxies=len(t)-1,
        results={label:metrics(group) for label,group in frame.groupby('comparison')},
        interpretation='Small shared local calibration sample, not a blind external ranking. The deliberately mismatched observable can have smaller median absolute error by accident; that does not validate its uncertainty model or physical semantics. No method winner is selected from this audit.')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
