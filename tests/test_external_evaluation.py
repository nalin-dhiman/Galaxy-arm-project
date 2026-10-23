"""Guard against false negative truth and outcome-based support selection."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from experiments.evaluate_next_external import spatial_metrics,support_decision


def test_unasked_workflow_never_becomes_arm_negative_truth():
    valid=np.ones((12,12),bool);probability=np.full(valid.shape,.9)
    # Even a supplied pixel array cannot manufacture truth without participation.
    score=spatial_metrics(probability,np.zeros(valid.shape),valid,0,.3)
    assert score['truth_pixels_scored']==0
    assert score['average_precision'] is None and score['brier'] is None
    assert score['candidate_area_fraction']==1.
    assert score['status']=='unknown_unasked_or_insufficient_spatial_annotations'


def test_consensus_negative_does_not_claim_physical_arm_absence():
    valid=np.ones((12,12),bool);probability=np.full(valid.shape,.8)
    score=spatial_metrics(probability,np.zeros(valid.shape),valid,15,.3)
    assert score['average_precision'] is None
    assert score['status']=='no_two_class_spatial_consensus'
    assert 'not proof of physical absence' in score['interpretation']


def test_fixed_vote_sensitivity_does_not_retune_prediction_threshold():
    valid=np.ones((12,12),bool);votes=np.zeros(valid.shape);votes[4:8]=3;votes[8:]=6
    probability=np.full(valid.shape,.1);probability[4:8]=.5;probability[8:]=.9
    primary=spatial_metrics(probability,votes,valid,15,.3,3)
    sensitivity=spatial_metrics(probability,votes,valid,15,.3,5)
    assert primary['average_precision']==1. and primary['f1']==1.
    assert np.isclose(sensitivity['f1'],2/3)
    assert primary['candidate_area_fraction']==sensitivity['candidate_area_fraction']


def test_support_abstention_uses_inputs_and_preserves_frozen_predictions():
    model=SimpleNamespace(features=['a','b'],training_min={'a':0.,'b':0.},training_max={'a':1.,'b':1.})
    features=pd.DataFrame({'a':[.5,np.nan,2.],'b':[.5,.5,.5]})
    predictions=pd.DataFrame({'predicted_catalog_redshift':[.03,.04,.05]});before=predictions.copy(deep=True)
    decisions=support_decision(model,features,predictions)
    assert decisions.suggested_abstention.tolist()==[False,True,True]
    assert decisions.missing_feature_names.tolist()==['','a','']
    assert decisions.outside_feature_names.tolist()==['','','a']
    assert_frame_equal(predictions,before)
