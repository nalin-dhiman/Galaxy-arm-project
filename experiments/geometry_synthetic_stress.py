#!/usr/bin/env python3
"""Known-geometry stress checks, kept separate from observed reference validation."""
from pathlib import Path
import sys,json,dataclasses
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from galaxy_lab.filtering import ScienceImage,DiskGeometry,radius_map
from galaxy_lab.arm_geometry import (ExtractionConfig,project_points,extract_centerlines,
 centerline_metrics,fit_log_spiral,measure_cross_section_width)
OUT=ROOT/'executed_next/geometry';SCALE=.3

def run():
    if (OUT/'synthetic_stress.json').exists():raise RuntimeError('Synthetic stress results exist; preserve prior execution')
    frozen=json.loads((OUT/'frozen_extractor.json').read_text());config=ExtractionConfig(**frozen['selected_configuration']);rng=np.random.default_rng(97231)
    yy,xx=np.indices((192,192));grid=np.c_[xx.ravel(),yy.ravel()];rows=[];examples=[]
    for inc in [0,30,60,70]:
        g=DiskGeometry((96,96),25,inc,2.5,23.);rr=radius_map((192,192),g,SCALE)
        theta=np.linspace(-1,2,1000);rad=8/SCALE*np.exp(np.tan(np.radians(23))*theta)
        curves=[project_points(np.c_[rad*np.cos(theta+angle),rad*np.sin(theta+angle)],g) for angle in [0,np.pi]]
        dist=cKDTree(np.concatenate(curves)).query(grid)[0].reshape(192,192)*SCALE
        arms=.5*np.exp(-.5*(dist/(1.2/2.354820045))**2)
        base=.2*np.exp(-rr/15)
        for psf in [.8,1.5,3.]:
            smooth=gaussian_filter(base+arms,psf/2.354820045/SCALE)
            for noise in [.01,.04]:
                im=ScienceImage(smooth+rng.normal(0,noise,smooth.shape),SCALE,psf_fwhm_arcsec=psf)
                pred,meta=extract_centerlines(im,g,config)
                metrics=centerline_metrics([p['points_xy'] for p in pred],curves,SCALE,psf)
                widths=[measure_cross_section_width(im,c,half_width_arcsec=max(4,2*psf)) for c in curves]
                good=[w['deconvolved_fwhm_arcsec'] for w in widths if w['deconvolved_fwhm_arcsec'] is not None]
                row={'kind':'known_spiral','inclination_deg':inc,'psf_fwhm_arcsec':psf,'noise_sigma':noise,**metrics,
                     'true_pitch_deg':23,'true_sky_normal_intrinsic_fwhm_arcsec':1.2,
                     'median_measured_deconvolved_width_arcsec':float(np.median(good)) if good else None,
                     'width_error_arcsec':float(np.median(good)-1.2) if good else None,
                     'analytic_truth_pitch_recovery_deg':fit_log_spiral(curves[0],g)['pitch_deg'],
                     'extraction':meta}
                rows.append(row)
                if inc in [0,60] and psf in [.8,3.] and noise==.04:examples.append((im,curves,pred,row))
    g=DiskGeometry((96,96),0,0,2.5,23.);rr=radius_map((192,192),g,SCALE)
    for kind in ['smooth','bar','ring','foreground_stars']:
        base=.2*np.exp(-rr/12)
        if kind=='bar':base+=.5*np.exp(-.5*((xx-96)/25)**2-.5*((yy-96)/3)**2)
        if kind=='ring':base+=.5*np.exp(-.5*((rr-12)/.7)**2)
        if kind=='foreground_stars':
            for x,y in [(70,90),(120,75),(110,125),(56,120)]:base+=2*np.exp(-.5*((xx-x)**2+(yy-y)**2))
        for psf in [.8,1.5,3.]:
            for noise in [.01,.04]:
                im=ScienceImage(gaussian_filter(base,psf/2.354820045/SCALE)+rng.normal(0,noise,base.shape),SCALE,psf_fwhm_arcsec=psf)
                pred,meta=extract_centerlines(im,g,config)
                rows.append({'kind':kind,'psf_fwhm_arcsec':psf,'noise_sigma':noise,'predicted_curve_count':len(pred),
                             'false_positive_any':bool(pred),'source_mask_supplied':False,'extraction':meta})
    spirals=[r for r in rows if r['kind']=='known_spiral'];controls=[r for r in rows if r['kind']!='known_spiral']
    summary={'synthetic_positive_cases':len(spirals),'synthetic_negative_cases':len(controls),
             'positive_mean_centerline_f1':float(np.mean([r['f1'] for r in spirals])),
             'negative_cases_with_any_false_curve':sum(r['false_positive_any'] for r in controls),
             'negative_case_false_curve_rate':float(np.mean([r['false_positive_any'] for r in controls])),
             'median_absolute_width_error_arcsec':float(np.median([abs(r['width_error_arcsec']) for r in spirals if r['width_error_arcsec'] is not None])),
             'width_measurement_uses_true_centerlines':True,
             'width_truth':'constant sky-normal Gaussian width; not disk-intrinsic physical width',
             'not_observational_accuracy':True}
    (OUT/'synthetic_stress.json').write_text(json.dumps({'summary':summary,'cases':rows},indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for ax,(im,curves,pred,row) in zip(axes.flat,examples):
        ax.imshow(im.data,origin='lower',cmap='gray',vmin=0,vmax=.6)
        for i,c in enumerate(curves):ax.plot(*c.T,color='cyan',lw=1.3,label='Exact injected centerline' if i==0 else None)
        for i,p in enumerate(pred):ax.plot(*p['points_xy'].T,color='orange',ls='--',lw=1,label='Frozen extractor' if i==0 else None)
        ax.set_title(f"inclination{row['inclination_deg']}deg, PSF{row['psf_fwhm_arcsec']}arcsec\nF1={row['f1']:.2f}, noise0.04");ax.set_xticks([]);ax.set_yticks([])
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2);fig.suptitle('Synthetic stress only: known geometry, Gaussian PSF and width assumptions');fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'synthetic_stress.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
