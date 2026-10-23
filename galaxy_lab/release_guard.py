"""Explicit acceptance/abstention for the frozen selected-population predictor.

These predeclared checks do not certify accuracy or remove population shift.
They keep estimates unavailable outside the stated observational/model scope.
"""
from pathlib import Path
import json
from html import escape


def screen_analysis(metadata,*,survey,population):
    reasons=[]
    if survey!='legacy-dr9':reasons.append('survey_not_validated_for_frozen_model')
    if population!='strong-resolved-spiral':reasons.append('population_not_validated_for_frozen_model')
    redshift=metadata.get('redshift')
    prediction=redshift.get('prediction',{}) if redshift else {}
    if not prediction:reasons.append('redshift_prediction_unavailable')
    else:
        if prediction.get('outside_training_feature_range',True):reasons.append('predictor_outside_training_range')
        if prediction.get('missing_predictors',1)>0:reasons.append('missing_observed_predictors')
        if prediction.get('status')!='experimental_selected_population_model':reasons.append('unrecognized_model_status')
    accepted=not reasons
    return dict(status='conditional_catalog_redshift' if accepted else 'abstain',
        reasons=reasons,declared_survey=survey,declared_population=population,
        accepted_redshift=prediction.get('predicted_catalog_redshift') if accepted else None,
        accepted_interval_lower=prediction.get('redshift_interval_lower') if accepted else None,
        accepted_interval_upper=prediction.get('redshift_interval_upper') if accepted else None,
        absolute_distance_status='not_identified_from_arm_geometry',
        viscosity_status='not_identified',
        interpretation='Passing these input/scope checks is not a guarantee of accuracy or calibrated external-population coverage. Survey/population are explicit caller declarations. Redshift and its residual interval do not measure geometric distance uncertainty.')


def write_screen(analysis_dir,*,survey,population):
    directory=Path(analysis_dir)
    result=screen_analysis(json.loads((directory/'analysis.json').read_text()),survey=survey,population=population)
    destination=directory/'release_decision.json'
    if destination.exists():raise FileExistsError('Preserve existing release decision')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_guard_report(directory,result)
    return result


def write_guard_report(directory,decision):
    """Show accepted values only; retain raw predictions behind diagnostics."""
    directory=Path(directory);path=directory/'release.html'
    if path.exists():raise FileExistsError('Preserve existing guarded report')
    if decision['status']=='abstain':
        message='Redshift estimate withheld.'
        reasons={
            'survey_not_validated_for_frozen_model':'The image survey is outside the evaluated scope.',
            'population_not_validated_for_frozen_model':'This galaxy population is outside the evaluated scope.',
            'redshift_prediction_unavailable':'Required redshift inputs or a model were not supplied.',
            'predictor_outside_training_range':'Observed predictors exceed the training range.',
            'missing_observed_predictors':'Required observed predictors are missing.',
            'unrecognized_model_status':'The model status cannot be verified.'}
        detail=' '.join(reasons.get(r,r) for r in decision['reasons'])
    else:
        message=f'Experimental catalog redshift: {decision["accepted_redshift"]:.4f}'
        detail=f'Residual-based interval: [{decision["accepted_interval_lower"]:.4f}, {decision["accepted_interval_upper"]:.4f}]. Passing input checks does not guarantee external-population accuracy.'
    html='<!doctype html><meta charset="utf-8"><title>Galaxy analysis result</title><style>body{font:18px/1.5 sans-serif;max-width:1100px;margin:40px auto;padding:20px}img{width:100%}</style>'
    html+='<h1>'+escape(message)+'</h1><p>'+escape(detail)+'</p>'
    html+='<p>Absolute distance from arm geometry and viscosity remain unmeasured. Candidate arm regions require inspection.</p>'
    if (directory/'analysis.png').exists():html+='<img src="analysis.png" alt="Science bands and candidate arm regions">'
    html+='<p><a href="analysis.html">Raw model diagnostics</a> · <a href="release_decision.json">Decision details</a></p>'
    path.write_text(html)
    return path
