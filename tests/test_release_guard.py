import json
from galaxy_lab.release_guard import screen_analysis,write_screen


def valid():
    return {'redshift':{'prediction':{'predicted_catalog_redshift':.03,'redshift_interval_lower':0,
        'redshift_interval_upper':.08,'outside_training_feature_range':False,'missing_predictors':0,
        'status':'experimental_selected_population_model'}}}


def test_scope_pass_is_conditional_not_distance():
    r=screen_analysis(valid(),survey='legacy-dr9',population='strong-resolved-spiral')
    assert r['accepted_redshift']==.03
    assert r['absolute_distance_status']=='not_identified_from_arm_geometry'


def test_unknown_population_or_missing_predictor_withholds_estimate():
    a=valid();a['redshift']['prediction']['missing_predictors']=1
    r=screen_analysis(a,survey='other-survey',population='unknown')
    assert r['status']=='abstain' and r['accepted_redshift'] is None
    assert len(r['reasons'])==3


def test_out_of_training_range_withholds_estimate():
    a=valid();a['redshift']['prediction']['outside_training_feature_range']=True
    r=screen_analysis(a,survey='legacy-dr9',population='strong-resolved-spiral')
    assert r['accepted_redshift'] is None


def test_abstaining_report_does_not_present_raw_prediction(tmp_path):
    (tmp_path/'analysis.json').write_text(json.dumps(valid()))
    write_screen(tmp_path,survey='unvalidated',population='strong-resolved-spiral')
    html=(tmp_path/'release.html').read_text()
    assert 'Redshift estimate withheld' in html
    assert '0.0300' not in html
    assert 'Raw model diagnostics' in html
