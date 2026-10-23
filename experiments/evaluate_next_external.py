"""Evaluate frozen arm/redshift artifacts on the preregistered fresh cohort.

No fit, tuning, calibration, artifact modification, or label-derived feature
input. Same-survey spatial/population validation, not independent-survey proof.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.metrics import average_precision_score
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.native import read_native_grz,annotation_center_and_stars
from galaxy_lab.filtering import DiskGeometry,display_transform
from galaxy_lab.segmentation import predict_arm_regions
from galaxy_lab.redshift import observable_frame,IMAGE_SOURCE,ARM
from galaxy_lab.distance import paired_group_bootstrap
from experiments.independent_filter_experiment import label_coordinates,map_labels

OUT=ROOT/'executed_next/external'
DATA=OUT/'data'
STRATA=('strong_spiral','weak_lowbar','weak_bar_proxy','low_vote_unknown')
SEED=20261024


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(x) for x in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write(path,value):
    path.write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n')


def verify_protocol():
    protocol=json.loads((OUT/'protocol.json').read_text())
    problems=[]
    for name,record in protocol['frozen_models'].items():
        if sha(ROOT/record['path'])!=record['sha256']:problems.append('model:'+name)
    for path,expected in protocol['inference_sources_sha256'].items():
        if sha(ROOT/path)!=expected:problems.append('source:'+path)
    if sha(DATA/'selected_targets.json')!=protocol['selection_sha256']:problems.append('selection')
    if problems:raise RuntimeError('Frozen protocol changed: '+','.join(problems))
    return protocol


def spatial_metrics(probability,votes,valid,n_classifications,threshold,vote_threshold=3):
    """Unasked/insufficient spatial tasks are unknown, never negative truth."""
    valid=np.asarray(valid,bool)&np.isfinite(probability)
    p=np.asarray(probability,float)[valid]
    out=dict(vote_threshold=vote_threshold,valid_prediction_pixels=int(valid.sum()),
             candidate_area_fraction=float(np.mean(p>=threshold)) if len(p) else None,
             score_mean=float(np.mean(p)) if len(p) else None,
             spatial_classifications=int(n_classifications),arm_ground_truth_feature_input=False)
    if n_classifications<10:
        return {**out,'status':'unknown_unasked_or_insufficient_spatial_annotations','truth_pixels_scored':0,
                'average_precision':None,'f1':None,'brier':None}
    if len(p)<100:
        return {**out,'status':'insufficient_valid_pixels','truth_pixels_scored':0,'average_precision':None,'f1':None,'brier':None}
    y=np.asarray(votes)[valid]>=vote_threshold;pred=p>=threshold
    positive=int(y.sum());tp=int(np.sum(pred&y));fp=int(np.sum(pred&~y));fn=int(np.sum(~pred&y))
    base={**out,'truth_pixels_scored':len(p),'positive_annotation_pixels':positive,
          'annotation_prevalence':float(y.mean()),'brier':float(np.mean((p-y)**2)),
          'true_positive_pixels':tp,'protocol_disagreement_positive_pixels':fp,'false_negative_pixels':fn}
    if not y.any() or y.all():
        return {**base,'status':'no_two_class_spatial_consensus','average_precision':None,'f1':None,
                'interpretation':'No AP defined; zero consensus is not proof of physical absence of arms'}
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return {**base,'status':'scored_conditional_annotation_agreement','average_precision':float(average_precision_score(y,p)),
            'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-30),
            'interpretation':'Agreement with incomplete human projected-region labels, not physical arm certainty'}


def support_decision(model,features,prediction):
    """A preregistered ancillary input-only abstention; never alter predictions."""
    rows=[]
    for index,row in features.iterrows():
        missing=[key for key in model.features if not np.isfinite(float(row[key]))]
        outside=[key for key in model.features if np.isfinite(float(row[key])) and
                 (float(row[key])<model.training_min[key] or float(row[key])>model.training_max[key])]
        rows.append(dict(suggested_abstention=bool(missing or outside),missing_feature_names=';'.join(missing),
                         outside_feature_names=';'.join(outside),
                         decision='suggested_abstention' if missing or outside else 'within_frozen_marginal_support',
                         prediction_preserved=True))
    return pd.DataFrame(rows,index=features.index)


def grouped_metric_interval(table,key):
    """Galaxy-macro point mean with whole-sky-group bootstrap uncertainty."""
    data=table[['sky_group',key]].dropna()
    if data.empty:return dict(mean=None,ci95_low=None,ci95_high=None,galaxies=0,groups=0)
    grouped=data.groupby('sky_group')[key].agg(['sum','count'])
    if len(grouped)<2:
        return dict(mean=float(data[key].mean()),ci95_low=None,ci95_high=None,galaxies=len(data),groups=len(grouped),status='insufficient independent groups')
    rng=np.random.default_rng(SEED);sampled=rng.integers(0,len(grouped),size=(10000,len(grouped)))
    means=grouped['sum'].to_numpy()[sampled].sum(axis=1)/grouped['count'].to_numpy()[sampled].sum(axis=1)
    return dict(mean=float(data[key].mean()),ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),
                galaxies=len(data),groups=len(grouped),resampling_unit='whole sky group; galaxy-macro mean',bootstrap_draws=10000,
                limitations='Conditional on frozen artifacts and selected cohort; no retraining or annotation-systematic uncertainty')


def redshift_metrics(frame,table,family,mask=None):
    if mask is None:mask=np.ones(len(frame),bool)
    mask=np.asarray(mask,bool)&np.isfinite(frame.spec_z.to_numpy(float))
    if not mask.any():return dict(status='no_eligible_rows',galaxies=0)
    y=frame.spec_z.to_numpy(float)[mask];p=table[family+'_predicted_catalog_redshift'].to_numpy(float)[mask]
    lo=table[family+'_redshift_interval_lower'].to_numpy(float)[mask];hi=table[family+'_redshift_interval_upper'].to_numpy(float)[mask]
    dz=p-y;norm=dz/(1+y);coverage=(lo<=y)&(y<=hi)
    groups=frame.sky_group.to_numpy()[mask]
    groupcover=pd.DataFrame(dict(group=groups,covered=coverage)).groupby('group').covered.all()
    return dict(status='evaluated_frozen_predictions',galaxies=int(mask.sum()),sky_groups=len(np.unique(groups)),
        mae_z=float(np.mean(abs(dz))),median_absolute_error_z=float(np.median(abs(dz))),mean_signed_error_z=float(np.mean(dz)),
        sigma_nmad_delta_z_over1pz=float(1.4826*np.median(abs(norm-np.median(norm)))),
        outlier_fraction_abs_dz_over1pz_gt0p15=float(np.mean(abs(norm)>.15)),
        galaxy_interval_coverage=float(np.mean(coverage)),whole_group_interval_coverage=float(groupcover.mean()),
        median_interval_width_z=float(np.median(hi-lo)),
        outside_training_feature_range_fraction=float(table[family+'_outside_training_feature_range'].to_numpy(bool)[mask].mean()),
        missing_predictor_fraction=float((table[family+'_missing_predictors'].to_numpy(int)[mask]>0).mean()),
        suggested_abstention_fraction=float(table[family+'_suggested_abstention'].to_numpy(bool)[mask].mean()))


def evaluate_objects(selected,annotations,natives,protocol):
    display_artifact=joblib.load(ROOT/protocol['frozen_models']['arm_display']['path'])
    raw_artifact=joblib.load(ROOT/protocol['frozen_models']['arm_raw_train_only']['path'])
    annot={r['id']:r for r in annotations['targets']};native={r['id']:r for r in natives['targets']}
    descriptor_rows=[];quality_rows=[];scores=[];failures=[];coverage=[];gallery={}
    (OUT/'predictions').mkdir(exist_ok=True)
    for number,target in enumerate(selected['targets'],1):
        identifier=target['id'];descriptor=dict(id=identifier,sample_group=target['sample_group'],sky_group=target['sky_group'],split='external')
        quality=dict(id=identifier);detail=dict(id=identifier,sample_group=target['sample_group'])
        try:
            n=native[identifier];a=annot[identifier]
            if n.get('status')!='available' or n.get('psf',{}).get('status')!='available' or a.get('status')!='available':
                raise ValueError('Missing frozen scientific input/PSF/independent reference marks')
            images,metadata=read_native_grz(ROOT/n['path'],ROOT/n['psf_path'],declared_unit='nanomaggy/pixel',
                target_sky=SkyCoord(target['ra_deg']*u.deg,target['dec_deg']*u.deg))
            # Verify inference selects exactly the frozen target footprint.
            for band in 'grz':
                if metadata['selected_hdus'][band]!={key:n['bands'][band][key] for key in ['image_hdu','invvar_hdu']}:
                    raise ValueError('Public reader and frozen footprint differ')
            quality.update(metadata['quality_features'])
            center,stars=annotation_center_and_stars(ROOT/a['path'],images['r'])
            for image in images.values():image.mask=~image.valid|stars
            geometry=DiskGeometry(center,0.,0.,3.,23.,'independent human center and foreground-star marks; fixed projected angular annulus')
            # These APIs receive no human arm mask, redshift or physical property.
            display=predict_arm_regions(display_artifact,images,geometry)
            raw=predict_arm_regions(raw_artifact,images,geometry)
            descriptor.update(raw['descriptors'],prediction_source='frozen_train_only_model_external',apparent_arm_features_available=1.)
            # Arm truth is opened ONLY after both predictions are finished.
            with fits.open(ROOT/a['path']) as hdus:
                from astropy.wcs import WCS
                xs,ys=label_coordinates(WCS(hdus[a['wcs_hdu']].header).celestial,images['r'].wcs,images['r'].data.shape)
                votes,label_valid=map_labels(hdus[a['spiral_hdu']].data,xs,ys)
            valid=display['valid']&label_valid
            if not np.array_equal(raw['valid'],display['valid']):raise ValueError('Frozen display/raw feature footprints differ')
            for vote_threshold in [3,5]:
                score=spatial_metrics(display['probability'],votes,valid,a['number_raw_spiral_classifications'],display_artifact['threshold'],vote_threshold)
                scores.append(dict(id=identifier,sample_group=target['sample_group'],sky_group=target['sky_group'],**score))
            detail.update(status='predicted',analysis_pixels=int(raw['valid'].sum()),label_overlap_pixels=int(valid.sum()),
                native_shape=list(images['r'].data.shape),selected_brick=metadata['selected_native_brick'],
                selected_hdus=metadata['selected_hdus'],spatial_classifications=a['number_raw_spiral_classifications'])
            header=images['r'].wcs.to_header();header['HISTORY']='Frozen external-cohort inference; no model tuning; human arm truth excluded from features'
            fits.writeto(OUT/'predictions'/f'{identifier}_arm_score.fits',display['probability'].astype(np.float32),header,overwrite=True)
            fits.writeto(OUT/'predictions'/f'{identifier}_valid.fits',display['valid'].astype(np.uint8),header,overwrite=True)
            if identifier in protocol['gallery_ids']:
                gallery[identifier]=dict(r=images['r'].data.copy(),score=display['probability'].copy(),valid=valid.copy(),votes=votes,
                    annotated=a['number_raw_spiral_classifications']>=10,sample_group=target['sample_group'],threshold=display_artifact['threshold'])
        except Exception as error:
            failure=dict(id=identifier,sample_group=target['sample_group'],error=f'{type(error).__name__}: {error}')
            failures.append(failure);detail.update(status='failed',error=failure['error'])
            descriptor.update(prediction_source='failed_fixed_input_or_feature_guards',apparent_arm_features_available=0.)
        for key in [*IMAGE_SOURCE,*ARM]:descriptor.setdefault(key,np.nan)
        for band in 'grz':
            quality.setdefault('psf_'+band+'_fwhm_arcsec',np.nan);quality.setdefault('apparent_noise_'+band+'_nanomaggy',np.nan)
        descriptor_rows.append(descriptor);quality_rows.append(quality);coverage.append(detail)
        if number%10==0 or number==len(selected['targets']):print(json.dumps(dict(predicted=number,total=len(selected['targets']),failures=len(failures))),flush=True)
    return pd.DataFrame(descriptor_rows),pd.DataFrame(quality_rows),pd.DataFrame(scores),coverage,failures,gallery


def plot_outputs(frame,predictions,scores,gallery,protocol):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/next_external_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(STRATA,['#24649b','#e09a22','#933da6','#666666']))
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    for axis,family in zip(axes,['P','PGI','PGIA']):
        for stratum in STRATA:
            use=frame.sample_group==stratum;x=frame.loc[use,'spec_z'].to_numpy();y=predictions.loc[use,family+'_predicted_catalog_redshift'].to_numpy()
            low=predictions.loc[use,family+'_redshift_interval_lower'].to_numpy();high=predictions.loc[use,family+'_redshift_interval_upper'].to_numpy()
            axis.errorbar(x,y,yerr=[y-low,high-y],fmt='o',markersize=3,alpha=.6,color=colors[stratum],linewidth=.6,label=stratum)
        limit=max(float(frame.spec_z.max()),float(predictions[family+'_predicted_catalog_redshift'].max()))*1.05
        axis.plot([0,limit],[0,limit],color='black',linewidth=.8);axis.set(xlabel='Inherited NSA catalog redshift',ylabel='Frozen prediction',title=family)
    axes[0].legend(fontsize=6);fig.suptitle('Fresh spatial/morphology cohort; unchanged models and90% intervals')
    fig.savefig(OUT/'external_redshift_validation.png',dpi=150);plt.close(fig)
    if not scores.empty:
        primary=scores[scores.vote_threshold==3]
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
        groups=[];names=[];areas=[]
        for stratum in STRATA:
            values=primary.loc[primary.sample_group==stratum,'average_precision'].dropna().to_numpy()
            groups.append(values if len(values) else [np.nan]);names.append(stratum+'\nAP n='+str(len(values)))
            areas.append(primary.loc[primary.sample_group==stratum,'candidate_area_fraction'].dropna().to_numpy())
        axes[0].boxplot(groups,tick_labels=names);axes[0].set(ylabel='Per-galaxy annotation-agreement AP',ylim=(0,1))
        axes[1].boxplot([a if len(a) else [np.nan] for a in areas],tick_labels=[s+'\nn='+str(len(a)) for s,a in zip(STRATA,areas)])
        axes[1].set(ylabel='Fixed-threshold candidate area fraction',ylim=(0,1));fig.suptitle('Unasked controls have unknown arm truth; candidate occupancy is descriptive')
        for ax in axes:ax.tick_params(axis='x',labelsize=6)
        fig.savefig(OUT/'external_arm_validation.png',dpi=150);plt.close(fig)
    if gallery:
        selected=[identifier for identifier in protocol['gallery_ids'] if identifier in gallery]
        fig,axes=plt.subplots(len(selected),4,figsize=(11,2.7*len(selected)),squeeze=False,constrained_layout=True)
        for row,identifier in enumerate(selected):
            item=gallery[identifier];valid=item['valid'];display=display_transform(item['r'],'asinh')
            axes[row,0].imshow(display,cmap='gray',origin='lower');axes[row,0].set_title(identifier+' '+item['sample_group'],fontsize=7)
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#dddddd')
            axes[row,1].imshow(np.where(valid,item['score'],np.nan),cmap=cmap,origin='lower',vmin=0,vmax=1);axes[row,1].set_title('Frozen annotation-agreement score',fontsize=7)
            if item['annotated']:
                axes[row,2].imshow(np.where(valid,item['votes']>=3,np.nan),cmap='Greens',origin='lower',vmin=0,vmax=1)
                axes[row,2].set_title('Independent >=3-vote region',fontsize=7)
            else:
                axes[row,2].set_facecolor('#dddddd');axes[row,2].text(.5,.5,'No spatial arm task\nTRUTH UNKNOWN',ha='center',va='center',transform=axes[row,2].transAxes,fontsize=8)
            axes[row,3].imshow(np.where(valid,item['score']>=item['threshold'],np.nan),cmap='Reds',origin='lower',vmin=0,vmax=1);axes[row,3].set_title('Unchanged candidate threshold',fontsize=7)
            for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle('Fixed gallery identities chosen before scoring; gray=no comparable pixels. Region labels are incomplete observations.')
        fig.savefig(OUT/'external_fixed_gallery.png',dpi=135);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-recorded-download-failures',action='store_true')
    args=parser.parse_args();protocol=verify_protocol();selected=json.loads((DATA/'selected_targets.json').read_text())
    annotations=json.loads((DATA/'annotations/manifest.json').read_text());natives=json.loads((DATA/'native_images/manifest.json').read_text())
    if annotations['pending'] or natives['pending']:raise RuntimeError('External acquisition still pending; cannot score a partial convenience subset')
    if (annotations['failures'] or natives['failures']) and not args.allow_recorded_download_failures:
        raise RuntimeError('Download failures retained; explicitly acknowledge them before evaluating every fixed identity')
    descriptors,quality,scores,coverage,failures,gallery=evaluate_objects(selected,annotations,natives,protocol)
    descriptors.to_csv(OUT/'external_image_descriptors.csv',index=False);quality.to_csv(OUT/'external_quality.csv',index=False)
    scores.to_csv(OUT/'external_arm_scores.csv',index=False);write(OUT/'coverage.json',coverage);write(OUT/'failures.json',failures)
    catalog=pd.read_csv(DATA/'matched_catalog.csv')
    metadata=pd.DataFrame(selected['targets'])[['id','sample_group']]
    catalog=catalog.merge(metadata,on='id',validate='one_to_one')
    order=catalog.id.tolist();descriptors=descriptors.set_index('id').reindex(order).reset_index();quality=quality.set_index('id').reindex(order).reset_index()
    features=observable_frame(catalog,descriptors,quality);features['apparent_arm_features_available']=descriptors.apparent_arm_features_available.to_numpy()
    features.to_csv(OUT/'external_observed_predictors.csv',index=False)
    predictions=catalog[['id','sky_group','sample_group','spec_z','drp3qual']].copy();models={}
    for family in ['P','PG','PGI','PGIA']:
        model=joblib.load(ROOT/protocol['frozen_models'][family]['path'])
        if model.metadata['distance_arm_model_sha256']!=protocol['frozen_models']['arm_raw_train_only']['sha256']:
            raise RuntimeError('Frozen redshift model/raw-arm artifact mismatch')
        result=model.predict(features);support=support_decision(model,features,result)
        for column in result:predictions[family+'_'+column]=result[column].to_numpy()
        for column in support:predictions[family+'_'+column]=support[column].to_numpy()
        models[family]=model
    predictions.to_csv(OUT/'external_redshift_predictions.csv',index=False)
    summaries={};paired={};arm_summary={};support_summary={}
    for stratum in [*STRATA,'pooled']:
        use=np.ones(len(catalog),bool) if stratum=='pooled' else catalog.sample_group.to_numpy()==stratum
        summaries[stratum]={family:redshift_metrics(catalog,predictions,family,use) for family in models}
        support_summary[stratum]={}
        for family in models:
            retained=use&~predictions[family+'_suggested_abstention'].to_numpy(bool)
            support_summary[stratum][family]=dict(retained=int(retained.sum()),total=int(use.sum()),metrics=redshift_metrics(catalog,predictions,family,retained))
        label_good=use&np.isfinite(catalog.spec_z.to_numpy(float))
        if label_good.any():
            y=catalog.spec_z.to_numpy()[label_good]
            paired[stratum]=paired_group_bootstrap(catalog.sky_group.to_numpy()[label_good],
                abs(predictions.PGI_predicted_catalog_redshift.to_numpy()[label_good]-y),
                abs(predictions.PGIA_predicted_catalog_redshift.to_numpy()[label_good]-y),seed=SEED,n_bootstrap=10000)
        if not scores.empty:
            subset=scores if stratum=='pooled' else scores[scores.sample_group==stratum]
            arm_summary[stratum]={}
            for threshold in [3,5]:
                table=subset[subset.vote_threshold==threshold];defined=table.average_precision.notna()
                arm_summary[stratum][str(threshold)]=dict(predicted_galaxies=len(table),ap_evaluable=int(defined.sum()),
                    unknown_spatial_truth=int((table.status=='unknown_unasked_or_insufficient_spatial_annotations').sum()),
                    no_two_class_consensus=int((table.status=='no_two_class_spatial_consensus').sum()),
                    macro_average_precision=float(table.loc[defined,'average_precision'].mean()) if defined.any() else None,
                    median_f1=float(table.loc[defined,'f1'].median()) if defined.any() else None,
                    average_precision_group_bootstrap=grouped_metric_interval(table,'average_precision'),
                    candidate_area_fraction_median=float(table.candidate_area_fraction.median()) if len(table) else None)
    subgroup_rows=[]
    for key,cuts in protocol['pretrained_quartile_cuts'].items():
        values=features[key].to_numpy(float)
        for name,select in [('below_old_train_Q25',values<cuts[0]),('old_train_interquartile',(values>=cuts[0])&(values<=cuts[1])),('above_old_train_Q75',values>cuts[1]),('missing',~np.isfinite(values))]:
            for family in models:subgroup_rows.append(dict(covariate=key,bin=name,family=family,**redshift_metrics(catalog,predictions,family,select)))
    pd.DataFrame(subgroup_rows).to_csv(OUT/'external_predefined_subgroups.csv',index=False)
    after=verify_protocol()
    summary=dict(status='executed_frozen_external_cohort_validation',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        survey_scope=selected['survey_scope'],frozen_protocol_sha256=sha(OUT/'protocol.json'),sample_size=len(catalog),
        sky_groups=catalog.sky_group.nunique(),cohort_counts=protocol['cohort_counts'],frozen_artifacts_unchanged=True,
        image_measurement_failures=len(failures),redshift=summaries,paired_PGI_minus_PGIA_mae_z=paired,
        strict_input_support=support_summary,arm_region_agreement=arm_summary,
        primary_redshift_stratum='strong_spiral',primary_redshift_contrast='PGI vsPGIA unchanged families, sky-group equal-weight bootstrap',
        primary_arm_stratum='strong_spiral',no_calibration_or_threshold_updates=True,
        limitations=['Same SDSS/MaNGA/Legacy surveys; no independently validated survey or infrared/UV image-domain transfer.',
                    'All prior sky tiles and galaxy IDs excluded; nearby large-scale structures can still correlate distinct sky tiles.',
                    'Human polygons provide incomplete projected arm-region labels, not pitch, physical separation, geometric distance or viscosity truth.',
                    'Low-vote objects without spatial workflow remain unknown, never negative physical arm labels.',
                    'Population/covariate shift invalidates automatic transfer of the original nominal90% interval guarantee; actual external coverage is measured.',
                    'Strict missing/range abstention is an ancillary conservative support diagnostic; all unchanged predictions remain in primary error/coverage tables.',
                    'NSA_Z is inherited heliocentric catalog redshift; this does not validate a geometric distance ruler.'])
    write(OUT/'summary.json',summary);plot_outputs(catalog,predictions,scores,gallery,protocol)
    primary=summary['redshift']['strong_spiral'];increment=summary['paired_PGI_minus_PGIA_mae_z']['strong_spiral']
    lower,upper=increment.get('ci95_low'),increment.get('ci95_high')
    if lower is not None and lower>0:
        conclusion='The frozen arm-augmented model has lower group-weighted error in this fresh selected strong-spiral cohort. This is a conditional statistical result, not a geometric ruler or causal evidence for fluid physics.'
    elif upper is not None and upper<0:
        conclusion='The frozen arm-augmented model has higher group-weighted error in this fresh selected strong-spiral cohort.'
    else:
        conclusion='The confidence interval includes zero: this fresh cohort does not establish a reliable incremental redshift benefit from arms.'
    lines=['FROZEN EXTERNAL-COHORT VALIDATION','',
        'This is a SAME-SURVEY fresh spatial/population test. It is not an independent-survey result.',
        'All192 prior galaxy IDs and all67 previous sky tiles were excluded before acquisition. Models, thresholds and interval radii were frozen before scoring.',
        f'Frozen sample: {len(catalog)} galaxies in {catalog.sky_group.nunique()} new sky tiles; counts {protocol["cohort_counts"]}.',
        f'Image measurement failures: {len(failures)}. Failed identities are retained; no replacements or accuracy-based exclusions.', '',
        'Primary fresh strong-spiral arm-region comparison:',
        str(summary['arm_region_agreement']['strong_spiral']['3']), '',
        'Primary fresh strong-spiral redshift comparison:']
    for family in ['P','PGI','PGIA']:
        metric=primary[family]
        lines.append(f'{family}: MAE_z={metric["mae_z"]:.6f}; frozen interval coverage={metric["galaxy_interval_coverage"]:.3f}; suggested abstention={metric["suggested_abstention_fraction"]:.3f}.')
    lines.extend([f'Group-weighted PGI-minus-PGIA absolute-error improvement={increment["mean_improvement"]:.6f};95% bootstrap interval=[{lower},{upper}].',conclusion,'',
        'Strict ancillary abstention is input-only: any missing or out-of-training-range predictor triggers suggested abstention. It never changes or hides the frozen prediction; all predictions remain in primary error tables.',
        'Weak and low-vote strata assess population/covariate shift. Unasked spatial arm tasks are unknown, never negative labels. Zero participating consensus is also not proof of physical arm absence.',
        'Arm metrics estimate agreement with incomplete projected human annotations, not pitch, physical separation, geometric distance or viscosity.',
        'NSA_Z is the inherited heliocentric catalog redshift. Original90% intervals have no automatic guarantee after covariate/population shift; external empirical coverage is reported.',
        'No new survey, infrared or UV transfer is established. No fitting, threshold tuning, recalibration, model-family reselection or fluid parameter inference occurred.',
        '', 'Machine-readable results: summary.json, external_redshift_predictions.csv, external_arm_scores.csv, coverage.json, failures.json.'])
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    write(OUT/'output_manifest.json',dict(files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.glob('*')) if p.is_file() and p.name not in ['output_manifest.json','evaluation.log']}))
    print(json.dumps(safe({key:summary[key] for key in ['status','sample_size','sky_groups','image_measurement_failures','cohort_counts']}),indent=2))


if __name__=='__main__':main()
