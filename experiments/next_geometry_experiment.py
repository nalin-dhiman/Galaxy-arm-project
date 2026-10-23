#!/usr/bin/env python3
"""Independent Galaxy Builder centerline validation; one frozen held-out execution."""
from pathlib import Path
import sys,json,hashlib,itertools,dataclasses
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from geometry_source_reader import load_data_only
from galaxy_lab.filtering import ScienceImage,DiskGeometry,radius_map
from galaxy_lab.arm_geometry import (geometry_from_builder,extract_centerlines,ExtractionConfig,
    centerline_metrics,fit_log_spiral,trace_bootstrap_pitch,measure_cross_section_width,
    deproject_points,project_points,resample_curve)
OUT=ROOT/'executed_next/geometry';SCALE=.396

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def psf_fwhm(kernel):
    p=np.maximum(np.asarray(kernel,float),0);yy,xx=np.indices(p.shape);p=p/p.sum();xc=(p*xx).sum();yc=(p*yy).sum()
    return float(2.354820045*np.sqrt((p*((xx-xc)**2+(yy-yc)**2)).sum()/2)*SCALE)

def load(row):
    a=load_data_only(ROOT/row['aggregate_path']);raw=json.loads((ROOT/row['image_path']).read_text())
    data=np.asarray(raw['imageData'],float);mask=np.asarray(raw['mask'],bool)
    if np.max(np.abs(np.asarray(data.shape)-np.asarray(a.spiral_pipeline.image_size)))>1.01:raise ValueError('Native image/trace coordinate mismatch exceeds1pixel historical rounding tolerance')
    if data.shape!=(raw['height'],raw['width']):raise ValueError('Native shape mismatch')
    fwhm=psf_fwhm(raw['psf']);im=ScienceImage(data,SCALE,unit='source-normalized linear SDSS r intensity',mask=mask,psf_fwhm_arcsec=fwhm,band='r')
    inner=max(2.,.12*min(data.shape)*SCALE);outer=.45*min(data.shape)*SCALE
    g=geometry_from_builder(a.centre_pos,a.phi,a.ba,inner,outer)
    refs=[];human=[]
    for arm in a.spiral_arms:
        curve=np.asarray(arm.reprojected_log_spiral,float)
        xy=deproject_points(curve,g);r=np.linalg.norm(xy,axis=1)*SCALE
        valid=(r>=inner)&(r<=outer)
        v=map_coordinates(im.valid.astype(float),[curve[:,1],curve[:,0]],order=0,mode='constant',cval=0)>.5
        valid &= v
        # Preserve separate contiguous segments; do not bridge masked star holes.
        segments=np.split(np.flatnonzero(valid),np.flatnonzero(np.diff(np.flatnonzero(valid))>1)+1)
        for indices in segments:
            if len(indices)>=12:
                segment=curve[indices]
                try:fit_log_spiral(segment,g)
                except ValueError:continue
                refs.append({'points_xy':segment,'published_pitch_deg':float(arm.pa),'author_flagged_bad':bool(arm.FLAGGED_AS_BAD)})
        human.append({'published_pitch_deg':float(arm.pa),'author_sigma_pitch_deg':float(arm.sigma_pa),
                      'author_flagged_bad':bool(arm.FLAGGED_AS_BAD),
                      **trace_bootstrap_pitch(arm.arms,g,n_bootstrap=300,seed=1023)})
    return im,g,refs,human

def evaluate(im,g,refs,config):
    pred,meta=extract_centerlines(im,g,config);p=[a['points_xy'] for a in pred];r=[a['points_xy'] for a in refs]
    result={**centerline_metrics(p,r,SCALE,im.psf_fwhm_arcsec),
      'half_psf_f1':centerline_metrics(p,r,SCALE,.5*im.psf_fwhm_arcsec)['f1'],
      'psf_fwhm_arcsec':im.psf_fwhm_arcsec,'extraction':meta}
    pairs=[]
    if pred and refs:
        cost=np.array([[centerline_metrics([pp['points_xy']],[rr['points_xy']],SCALE,im.psf_fwhm_arcsec)['symmetric_median_arcsec'] for rr in refs] for pp in pred])
        pi,ri=linear_sum_assignment(cost)
        for i,j in zip(pi,ri):
            if cost[i,j]>2*im.psf_fwhm_arcsec:continue
            try:fit=fit_log_spiral(pred[i]['points_xy'],g)
            except ValueError:continue
            pairs.append({'prediction_index':int(i),'reference_index':int(j),'median_distance_arcsec':float(cost[i,j]),
                'predicted_pitch_deg':fit['pitch_deg'],'reference_pitch_deg':refs[j]['published_pitch_deg'],
                'absolute_pitch_error_deg':abs(fit['pitch_deg']-refs[j]['published_pitch_deg'])})
    result['matched_segments']=pairs
    result['median_matched_pitch_error_deg']=float(np.median([p['absolute_pitch_error_deg'] for p in pairs])) if pairs else None
    return result,pred

def interval(values,seed=1023):
    v=np.asarray(values,float);rng=np.random.default_rng(seed);b=np.mean(rng.choice(v,(5000,len(v)),replace=True),axis=1)
    return {'mean':float(v.mean()),'interval95':np.quantile(b,[.025,.975]).tolist(),'n_galaxies':len(v),'resampling_unit':'whole galaxy'}

def run():
    if (OUT/'heldout_results.json').exists():raise RuntimeError('Frozen held-out results exist; refuse rerunning or retuning')
    selection=json.loads((OUT/'selection.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    cache={r['id']:load(r) for r in selection['targets']}
    bank=[]
    for spec in protocol['parameter_bank']:
        config=ExtractionConfig(**spec);values=[]
        for row in selection['targets']:
            if row['split']!='development':continue
            im,g,refs,human=cache[row['id']];res,pred=evaluate(im,g,refs,config);values.append(res['f1'])
        bank.append({'config':dataclasses.asdict(config),'development_mean_f1':float(np.mean(values)),'development_f1':values})
        print('development',spec,'F1',np.mean(values),flush=True)
    best=max(range(len(bank)),key=lambda i:bank[i]['development_mean_f1']);config=ExtractionConfig(**bank[best]['config'])
    frozen={'selected_configuration':dataclasses.asdict(config),'bank':bank,'protocol_sha256':sha(OUT/'protocol.json'),
      'selection_sha256':sha(OUT/'selection.json'),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'galaxy_lab/arm_geometry.py',ROOT/'experiments/geometry_source_reader.py']},
      'selection_completed_before_any_test_score':True,'not_the_frozen_supervised_model':True}
    dump(OUT/'frozen_extractor.json',frozen)
    results=[];robustness=[];humanrows=[];widthrows=[];gallery=[]
    for row in selection['targets']:
        im,g,refs,human=cache[row['id']];res,pred=evaluate(im,g,refs,config)
        results.append({'id':row['id'],'split':row['split'],**res})
        for i,h in enumerate(human):humanrows.append({'id':row['id'],'split':row['split'],'arm_index':i,**h})
        for i,ref in enumerate(refs):
            w=measure_cross_section_width(im,ref['points_xy'],half_width_arcsec=max(4.,2*im.psf_fwhm_arcsec))
            curve=resample_curve(ref['points_xy'],128);length=np.linalg.norm(np.diff(curve,axis=0),axis=1).sum()*SCALE
            nblocks=max(2,min(8,int(length/(2*im.psf_fwhm_arcsec))))
            widths=[]
            for block in np.array_split(curve,nblocks):
                ww=measure_cross_section_width(im,block,half_width_arcsec=max(4.,2*im.psf_fwhm_arcsec))
                if ww['observed_fwhm_arcsec'] is not None:widths.append(ww['observed_fwhm_arcsec'])
            w.update({'id':row['id'],'split':row['split'],'reference_segment':i,'block_count':len(widths),
                      'contiguous_block_width_16_84_arcsec':np.quantile(widths,[.16,.84]).tolist() if widths else None,
                      'block_spread_is_confidence_interval':False})
            widthrows.append(w)
        if row['split']=='test':
            for factor in [1.5,2.]:
                sigma=im.psf_fwhm_arcsec*np.sqrt(factor**2-1)/2.354820045/SCALE
                valid=im.valid;den=gaussian_filter(valid.astype(float),sigma);data=gaussian_filter(np.where(valid,im.data,0),sigma)/np.maximum(den,1e-10)
                blurred=ScienceImage(data,SCALE,mask=(~valid)|(den<.95),psf_fwhm_arcsec=im.psf_fwhm_arcsec*factor)
                rr,pp=evaluate(blurred,g,refs,config)
                # Fixed original PSF tolerance keeps localization scores comparable.
                fixed=centerline_metrics([p['points_xy'] for p in pp],[r['points_xy'] for r in refs],SCALE,im.psf_fwhm_arcsec)
                robustness.append({'id':row['id'],'perturbation':f'PSF_x{factor}',**rr,'f1_at_original_psf_tolerance':fixed['f1']})
            for key,delta in [('pa_deg',-5),('pa_deg',5),('inclination_deg',-5),('inclination_deg',5)]:
                value=getattr(g,key)+delta
                if key=='inclination_deg':value=np.clip(value,0,79)
                perturbed=dataclasses.replace(g,**{key:value});rr,pp=evaluate(im,perturbed,refs,config)
                robustness.append({'id':row['id'],'perturbation':key+f'{delta:+d}',**rr,'f1_at_original_psf_tolerance':rr['f1']})
            if len(gallery)<6:gallery.append((row,im,g,refs,pred))
        print('scored',row['id'],row['split'],'F1',round(res['f1'],3),'refs',len(refs),'pred',len(pred),flush=True)
    test=[r for r in results if r['split']=='test'];pitches=[p['absolute_pitch_error_deg'] for r in test for p in r['matched_segments']]
    summary={'test_centerline_f1':interval([r['f1'] for r in test]),'test_half_psf_f1':interval([r['half_psf_f1'] for r in test]),
      'test_mean_precision':float(np.mean([r['precision'] for r in test])),'test_mean_recall':float(np.mean([r['recall'] for r in test])),
      'test_predicted_segments':sum(r['predicted_curve_count'] for r in test),'test_reference_segments':sum(r['reference_curve_count'] for r in test),
      'test_matched_segment_count':len(pitches),'test_matched_median_pitch_error_deg':float(np.median(pitches)) if pitches else None,
      'test_galaxies_with_no_predictions':sum(r['predicted_curve_count']==0 for r in test),
      'test_galaxies_with_no_reference_in_annulus':sum(r['reference_curve_count']==0 for r in test),
      'acceptance_mean_f1_passed':float(np.mean([r['f1'] for r in test]))>=.7,
      'independent_observed_width_accuracy_established':False,'distance_accuracy_established':False,
      'reference_caveats':'author-cleaned clustered human freehand centerlines and fitted logarithmic reference; disk geometry also independent humanfit; high-quality selected lowinclination spirals, small12testsample',
      'frozen_sha256':sha(OUT/'frozen_extractor.json')}
    dump(OUT/'heldout_results.json',{'summary':summary,'objects':results});dump(OUT/'robustness.json',robustness)
    dump(OUT/'human_trace_uncertainty.json',humanrows);dump(OUT/'conditional_width_measurements.json',widthrows)
    render(gallery)
    print(json.dumps(summary,indent=2),flush=True)

def render(gallery):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    for ax,(row,im,g,refs,pred) in zip(axes.flat,gallery):
        v=im.data[im.valid];vmin,vmax=np.quantile(v,[.02,.995]);ax.imshow(np.arcsinh((im.data-vmin)/max(vmax-vmin,1e-8)*5),origin='lower',cmap='gray')
        for i,r in enumerate(refs):ax.plot(*r['points_xy'].T,color='#00e5ff',lw=2,label='Independent human-fit centerline' if i==0 else None)
        for i,p in enumerate(pred):ax.plot(*p['points_xy'].T,color='#ffb000',lw=1.5,ls='--',label='Image-only template candidate' if i==0 else None)
        ax.set_title(row['id']+' held out');ax.set_xlim(0,im.data.shape[1]);ax.set_ylim(0,im.data.shape[0]);ax.set_xticks([]);ax.set_yticks([])
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.suptitle('Galaxy Builder native SDSS r: angular centerline validation, fixed first six test objects\nHuman disk geometry supplied; template candidates can miss arms or follow unrelated light')
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'heldout_centerlines.png',dpi=160);plt.close(fig)
if __name__=='__main__':run()
