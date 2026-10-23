#!/usr/bin/env python3
"""Published HI-width acquisition, error-aware distances and method checks.

Local LOO calibrators were used in the preceding stage; this is an uncertainty
and observable-definition audit, not a newly blinded precision-distance test.
"""
from pathlib import Path
import sys
import json
import os
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from btfr_distance_experiment import read_table
from galaxy_lab.erroraware_btfr import ErrorAwareBTFR
OUT=ROOT/'executed_next/kinematics'
LN10=np.log(10.)


def table():
    names=['galaxy','logmb','e_logmb','inc2019','e_inc2019','vf2019','e_vf2019',
        'v2exp','e_v2exp','v2eff','e_v2eff','vmax','e_vmax','wp20','e_wp20','wm50','e_wm50','wm50c','e_wm50c']
    rows=[]
    for line in (OUT/'data/BTFR_Lelli2019.mrt').read_text().splitlines():
        a=line.split()
        if len(a)!=len(names):continue
        try:row=[a[0]]+[float(v) for v in a[1:]]
        except ValueError:continue
        rows.append(dict(zip(names,row)))
    t=pd.DataFrame(rows)
    if len(t)!=153 or t.galaxy.duplicated().any():raise ValueError('Expected153 quality-selected published rows')
    merged=t.merge(read_table(),on='galaxy',validate='one_to_one')
    d=merged.distance_mpc.to_numpy();ed=merged.distance_error_mpc.to_numpy()
    merged['flux_proxy']=10**merged.logmb.to_numpy()/d**2
    variance=merged.e_logmb.to_numpy()**2-(2*ed/(d*LN10))**2
    merged['non_distance_variance_before_floor']=variance
    merged['log_flux_error']=np.sqrt(np.maximum(.02**2,variance))
    return merged


def fit(frame,velocity='vf2019',error='e_vf2019',definition='HI_Vflat_kms',**kwargs):
    return ErrorAwareBTFR.fit(frame[velocity].to_numpy(),frame.logmb.to_numpy(),
        frame[error].to_numpy()/frame[velocity].to_numpy()/LN10,frame.e_logmb.to_numpy(),
        velocity_definition=definition,**kwargs)


def predict(model,frame,velocity='vf2019',error='e_vf2019',definition='HI_Vflat_kms'):
    rows=[]
    for k,(_,row) in enumerate(frame.iterrows()):
        p=model.predict(row[velocity],row.flux_proxy,row[error]/row[velocity]/LN10,
            row.log_flux_error,velocity_definition=definition,seed=20261023+k)
        p.update(galaxy=row.galaxy,reference_mpc=row.distance_mpc,
            reference_error_mpc=row.distance_error_mpc,reference_method=int(row.distance_method))
        rows.append(p)
    return pd.DataFrame(rows)


def metrics(f):
    if not len(f):return dict(galaxies=0)
    frac=f.distance_mpc.to_numpy()/f.reference_mpc.to_numpy()-1
    cover=(f.reference_mpc>=f.model_interval_90_lower_mpc)&(f.reference_mpc<=f.model_interval_90_upper_mpc)
    return dict(galaxies=len(f),median_absolute_fractional_error=float(np.median(abs(frac))),
        mean_absolute_fractional_error=float(np.mean(abs(frac))),median_signed_fractional_error=float(np.median(frac)),
        model_interval_90_reference_point_coverage=float(cover.mean()),
        median_interval_fractional_width=float(np.median((f.model_interval_90_upper_mpc-f.model_interval_90_lower_mpc)/f.distance_mpc)),
        velocity_extrapolations=int(f.outside_velocity_calibration_range.sum()))


def main():
    marker=OUT/'btfr_summary.json'
    if marker.exists():raise SystemExit('Existing evaluated run preserved; use a new explicit version to repeat')
    t=table();t.to_csv(OUT/'btfr_matched_data.csv',index=False)
    keep=t.distance_method.isin([2,3])&(t.vf2019>0)&(t.inclination_deg>=30)&(t.quality<=2)
    calibrators=t.loc[keep].reset_index(drop=True)
    calibrators.to_csv(OUT/'btfr_calibrators.csv',index=False)
    models=OUT/'models';models.mkdir(exist_ok=True)
    loo=[];ols=[];posteriors=[]
    for i,row in calibrators.iterrows():
        train=calibrators.drop(i);m=fit(train);p=predict(m,calibrators.iloc[[i]])
        p['calibration_galaxies']=len(train);loo.append(p);posteriors.append(dict(galaxy=row.galaxy,**m.summarize()))
        b,a=np.polyfit(np.log10(train.vf2019.to_numpy()/100),train.logmb,1)
        ols.append(dict(galaxy=row.galaxy,reference_mpc=row.distance_mpc,
            distance_mpc=float(np.sqrt(10**(a+b*np.log10(row.vf2019/100))/row.flux_proxy))))
    loo=pd.concat(loo,ignore_index=True);loo.to_csv(OUT/'btfr_loo_predictions.csv',index=False)
    pd.DataFrame(ols).to_csv(OUT/'btfr_same_data_ols.csv',index=False)
    (OUT/'btfr_loo_fit_diagnostics.json').write_text(json.dumps(posteriors,indent=2)+'\n')
    allmodel=fit(calibrators);allmodel.save(models/'HI_Vflat_posterior.npz')
    external=t.loc[(t.distance_method.isin([4,5]))&(t.vf2019>0)&(t.quality<=2)&(t.inclination_deg>=30)]
    external_pred=predict(allmodel,external);external_pred.to_csv(OUT/'btfr_method_holdouts.csv',index=False)
    # A separate HI-width relation is required; the table already corrects inclination.
    widthcal=calibrators.loc[(calibrators.wm50>0)&(calibrators.e_wm50>0)].copy()
    widthcal['half_width']=widthcal.wm50/2;widthcal['e_half_width']=widthcal.e_wm50/2
    widthrows=[]
    for i,row in widthcal.iterrows():
        subset=widthcal.drop(i)
        m=fit(subset,'half_width','e_half_width','HI_Wm50_over2_kms')
        right=predict(m,widthcal.loc[[i]],'half_width','e_half_width','HI_Wm50_over2_kms')
        # Deliberately wrong physical assumption, plainly labelled as a control.
        # Bypass API semantic refusal only by declaring the substituted input Vf.
        vm=fit(calibrators.drop(i))
        wrong=predict(vm,widthcal.loc[[i]],'half_width','e_half_width','HI_Vflat_kms')
        right['deliberately_wrong_velocity_substitution_mpc']=wrong.distance_mpc.iloc[0]
        right['measured_width_over2_kms']=row.half_width;right['measured_vflat_kms']=row.vf2019
        widthrows.append(right)
    widths=pd.concat(widthrows,ignore_index=True);widths.to_csv(OUT/'btfr_width_definition_test.csv',index=False)
    wmodel=fit(widthcal,'half_width','e_half_width','HI_Wm50_over2_kms');wmodel.save(models/'HI_Wm50_over2_posterior.npz')
    trgb=calibrators[calibrators.distance_method==2];cep=calibrators[calibrators.distance_method==3]
    transfer=predict(fit(trgb),cep);transfer.to_csv(OUT/'btfr_trgb_to_cepheid.csv',index=False)
    sensitivity=[]
    for label,sub,options in [('scatter_prior0p15',calibrators,dict(scatter_prior_scale=.15)),
        ('scatter_prior0p6',calibrators,dict(scatter_prior_scale=.6)),
        ('double_grid',calibrators,dict(grid_size=(401,321))),
        ('inclination45',calibrators[calibrators.inclination_deg>=45],{})]:
        m=fit(sub,**options);sensitivity.append(dict(scenario=label,**m.summarize()))
    widthwrong=widths.deliberately_wrong_velocity_substitution_mpc/widths.reference_mpc-1
    olsfrac=np.array([p['distance_mpc']/p['reference_mpc']-1 for p in ols])
    summary=dict(status='completed_conditional_uncertainty_audit',published_rows=len(t),calibration_galaxies=len(calibrators),
        same_calibrators_as_prior_stage=True,loo=metrics(loo),
        same_updated_data_ols_median_absolute_fractional_error=float(np.median(abs(olsfrac))),
        fitted_posterior=allmodel.summarize(),
        independent_method_holdouts={str(int(method)):metrics(external_pred[external_pred.reference_method==method]) for method in sorted(external_pred.reference_method.unique())},
        trgb_calibration_to_cepheid=metrics(transfer),
        correctly_calibrated_width=metrics(widths),
        deliberately_wrong_vflat_substitution=dict(galaxies=len(widths),median_absolute_fractional_error=float(np.median(abs(widthwrong))),median_signed_fractional_error=float(np.median(widthwrong))),
        non_distance_mass_variance_floor_cases=int(np.sum(t.non_distance_variance_before_floor<.02**2)),
        sensitivities=sensitivity,source_sha256=hashlib.sha256((OUT/'data/BTFR_Lelli2019.mrt').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((OUT/'btfr_protocol.json').read_bytes()).hexdigest(),
        limitations=['Same local calibrators previously studied; no new blinded geometric-distance precision claim.',
            'Shared distance anchors and galaxy environments are not a fitted covariance model; global0.05mag is an assumed sensitivity.',
            'UrsaMajor method4 labels use a common18Mpc group scale, so galaxies are not independent individual-distance references.',
            'Only2 supernova-method objects and3 Cepheids: method checks are small, not population calibration.',
            'Non-distance flux errors subtract the catalog distance variance with a stated0.02dex floor; rounded/correlated catalog errors limit this decomposition.',
            'Vertical Gaussian likelihood and flat latent-logV prior do not model sample selection or a complete stellar-population mass model.',
            'Model intervals have no independent coverage calibration; inclusion of a noisy reference point is not proof of true-distance coverage.',
            'Vflat and HI-width observables must be separately calibrated; optical rotation alone is not either observable.'])
    (OUT/'btfr_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/next_galaxy_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True)
    error=np.stack([loo.distance_mpc-loo.model_interval_90_lower_mpc,loo.model_interval_90_upper_mpc-loo.distance_mpc])
    axes[0].errorbar(loo.reference_mpc,loo.distance_mpc,yerr=error,fmt='o',alpha=.7,markersize=4)
    axes[0].plot([1,60],[1,60],'k--');axes[0].set(xscale='log',yscale='log',xlabel='TRGB/Cepheid reference (Mpc)',ylabel='LOO estimate + model90% interval (Mpc)')
    for method,label in [(4,'Shared UrsaMajor group'),(5,'Supernova method')]:
        group=external_pred[external_pred.reference_method==method]
        axes[1].scatter(group.reference_mpc,group.distance_mpc,label=label)
    axes[1].plot([10,30],[10,30],'k--');axes[1].set(xlabel='Different-method reference (Mpc)',ylabel='Fixed calibration prediction (Mpc)');axes[1].legend(fontsize=7)
    axes[2].scatter(widths.reference_mpc,widths.distance_mpc,label='Own width calibration')
    axes[2].scatter(widths.reference_mpc,widths.deliberately_wrong_velocity_substitution_mpc,label='Wrong Vf substitution',marker='x')
    axes[2].plot([1,60],[1,60],'k--');axes[2].set(xlabel='Independent reference (Mpc)',ylabel='HI-width estimate (Mpc)');axes[2].legend(fontsize=7)
    fig.suptitle('Distance uncertainty + observable-definition audit; conditional models, not image-only distance')
    fig.savefig(OUT/'btfr_uncertainty_validation.png',dpi=160);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('calibration_galaxies','loo','same_updated_data_ols_median_absolute_fractional_error','independent_method_holdouts','correctly_calibrated_width','deliberately_wrong_vflat_substitution')},indent=2))


if __name__=='__main__':main()
