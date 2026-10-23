#!/usr/bin/env python3
"""Train/freeze/calibrate one compact supervised human-arm-region predictor.

New galaxies exclude the previous 32. Sky groups, not pixels, separate all
splits. Hyperparameters use grouped cross-validation on TRAIN GALAXIES ONLY.
Calibration labels fit a monotone probability correction and a single binary
threshold. The final test is loaded/scored once, after frozen.json is written.
"""
from __future__ import annotations
import hashlib
import json
import argparse
import os
from pathlib import Path
from importlib.metadata import version
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/galaxy_lab_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.filtering import (ScienceImage, DiskGeometry, radius_map, match_gaussian_psf,
    display_transform, radial_residual)
from galaxy_lab.segmentation import (arm_features, probability_descriptors,
    predict_feature_probabilities, FEATURE_VERSION)
from experiments.independent_filter_experiment import (label_coordinates,
    map_labels, save_csv, paired_interval)

OUT = ROOT / 'executed_solid/segmentation'
DATA = ROOT / 'executed_solid/data'
SEED = 20261023
SCALES = (.4, .8, 1.6)
MAX_CLASS_PIXELS = 400
CALIBRATION_PIXELS_PER_GALAXY = 1500
CANDIDATES = ('brightness_radius_logistic', 'brightness_radius_extra_trees_depth8',
              'brightness_radius_extra_trees_depth12', 'full_logistic_c0.1',
              'full_logistic_c1', 'extra_trees_depth8', 'extra_trees_depth12')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n')


def mean_or_none(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def stable_seed(identifier):
    return int(hashlib.sha256(str(identifier).encode()).hexdigest()[:8], 16)


def read_manifest_native(native):
    """Honor audited explicit target-brick HDUs; never pick the last band HDU."""
    records = native.get('bands')
    if not isinstance(records, dict) or not all(b in records for b in 'grz'):
        raise ValueError('Audited per-band target-brick HDU indices are required')
    images = {}
    with fits.open(ROOT / native['path'], memmap=False) as hdus:
        bricks = []
        for band in 'grz':
            record = records[band]
            image_hdu, ivar_hdu = hdus[record['image_hdu']], hdus[record['invvar_hdu']]
            if str(image_hdu.header.get('BAND','')).strip() != band or str(ivar_hdu.header.get('BAND','')).strip() != band:
                raise ValueError('Manifest HDU band does not match native header')
            if image_hdu.header.get('IMAGETYP') != 'image' or ivar_hdu.header.get('IMAGETYP') != 'invvar':
                raise ValueError('Manifest image/inverse-variance HDU type mismatch')
            data, ivar = np.array(image_hdu.data,float), np.array(ivar_hdu.data,float)
            if data.ndim != 2 or data.shape != ivar.shape:
                raise ValueError('Manifest-selected image/ivar shapes differ')
            brick, ivbrick = image_hdu.header.get('BRICK'), ivar_hdu.header.get('BRICK')
            if brick != ivbrick:
                raise ValueError('Manifest-selected image and inverse variance belong to different bricks')
            if record.get('brick') is not None and record['brick'] != brick:
                raise ValueError('Manifest brick name differs from selected native header')
            bricks.append(brick)
            wcs, iwcs = WCS(image_hdu.header).celestial, WCS(ivar_hdu.header).celestial
            if not wcs.has_celestial or not iwcs.has_celestial:
                raise ValueError('Native celestial WCS missing')
            scales = np.asarray(proj_plane_pixel_scales(wcs))*3600
            if not np.isfinite(scales).all() or np.min(scales)<=0 or max(scales)/min(scales)>1.01:
                raise ValueError('Native pixels must have near-isotropic positive angular sampling')
            scale = float(np.mean(scales))
            if 'ra_deg' in native and 'dec_deg' in native:
                target_x,target_y=wcs.world_to_pixel_values(native['ra_deg'],native['dec_deg'])
                if not np.isfinite([target_x,target_y]).all() or not (-.5 <= target_x < data.shape[1]-.5 and -.5 <= target_y < data.shape[0]-.5):
                    raise ValueError('Manifest-selected native brick does not contain the requested galaxy center')
            xx, yy = np.meshgrid(np.linspace(0,data.shape[1]-1,3),np.linspace(0,data.shape[0]-1,3))
            offsets = wcs.pixel_to_world(xx,yy).separation(iwcs.pixel_to_world(xx,yy)).arcsec
            if not np.isfinite(offsets).all() or np.max(offsets)>.001*scale:
                raise ValueError('Native image and inverse variance WCS differ')
            valid = np.isfinite(data)&np.isfinite(ivar)&(ivar>0)
            variance = np.divide(1.,ivar,out=np.full(data.shape,np.nan),where=valid)
            images[band] = ScienceImage(data,scale,unit='nanomaggy/pixel',mask=~valid,
                variance=variance,wcs=wcs,band=band,pixel_value_kind='flux_per_pixel',
                psf_fwhm_arcsec=record.get('psf_fwhm_arcsec'),metadata={
                    'native_hdu':record['image_hdu'],'inverse_variance_hdu':record['invvar_hdu'],
                    'brick':brick,'unit_source':native.get('unit_source_url'),
                    'no_science_reprojection':True})
        if len(set(bricks)) != 1:
            raise ValueError('Manifest-selected g/r/z planes belong to different bricks')
    reference = images['r']
    xx, yy = np.meshgrid(np.linspace(0,reference.data.shape[1]-1,3),np.linspace(0,reference.data.shape[0]-1,3))
    for image in images.values():
        if image.data.shape != reference.data.shape:
            raise ValueError('Manifest-selected bands have different native shapes')
        offset=reference.wcs.pixel_to_world(xx,yy).separation(image.wcs.pixel_to_world(xx,yy)).arcsec
        if not np.isfinite(offset).all() or np.max(offset)>.001*reference.pixel_scale_arcsec:
            raise ValueError('Manifest-selected band WCS solutions differ')
    return images


def load_object(annotation, native):
    """Fast image features + nearest independent labels, no filter-bank sweep."""
    images = read_manifest_native(native)
    reference = images['r']
    with fits.open(ROOT / annotation['path']) as hdus:
        awcs = WCS(hdus[annotation['wcs_hdu']].header).celestial
        xs, ys = label_coordinates(awcs, reference.wcs, reference.data.shape)
        votes, label_valid = map_labels(hdus[annotation['spiral_hdu']].data, xs, ys)
        stars, _ = map_labels(hdus[annotation['stars_hdu']].data, xs, ys)
        center_sky = awcs.pixel_to_world(annotation['center_x'], annotation['center_y'])
        center = tuple(float(v) for v in reference.wcs.world_to_pixel(center_sky))
    for image in images.values():
        image.mask = ~image.valid | (stars > 0)
    input_psfs = {band: im.psf_fwhm_arcsec for band, im in images.items()}
    if any(p is None or not np.isfinite(p) for p in input_psfs.values()):
        raise ValueError('Missing local PSF model; no invented matching width')
    target_psf = max(input_psfs.values())
    images = {b: match_gaussian_psf(im, target_psf) for b, im in images.items()}
    geometry = DiskGeometry(center, 0, 0, 3, 23, 'independent human center; fixed circular sky annulus')
    image_features = arm_features(images, geometry, SCALES)
    scoring_selector = label_valid[image_features.valid]
    features = image_features.restrict(label_valid)
    if features.valid.sum() < 100:
        raise ValueError('Fewer than 100 common labeled feature pixels')
    rr = radius_map(reference.data.shape, geometry, reference.pixel_scale_arcsec)
    geometric = (rr >= 3) & (rr <= 23) & label_valid
    radial, _ = radial_residual(images['g'], geometry)
    meta = {'id': annotation['id'], 'split': annotation['split'],
        'sky_group': annotation['sky_group'], 'valid_pixels': int(features.valid.sum()),
        'image_feature_valid_pixels': int(image_features.valid.sum()),
        'image_geometric_pixels': int(((rr >= 3) & (rr <= 23)).sum()),
        'geometric_pixels': int(geometric.sum()),
        'coverage_fraction': float(features.valid.sum() / geometric.sum()) if geometric.any() else None,
        'positive_pixels_ge3': int(((votes >= 3) & features.valid).sum()),
        'positive_pixels_ge5': int(((votes >= 5) & features.valid).sum()),
        'input_psf_fwhm_arcsec': input_psfs, 'target_psf_fwhm_arcsec': target_psf,
        'star_mask_pixels': int(((stars > 0) & geometric).sum()),
        'native_file': native['path'], 'annotation_file': annotation['path']}
    return {'annotation': annotation, 'native': native, 'features': features, 'image_features': image_features,
        'scoring_selector': scoring_selector,
        'y3': (votes >= 3)[features.valid], 'y5': (votes >= 5)[features.valid],
        'raw_g': images['g'].data[features.valid],
        'radial_g': (images['g'].data - radial)[features.valid],
        'metadata': meta, 'images': images, 'geometry': geometry, 'votes': votes}


def candidate(name, names):
    if name.startswith('brightness_radius'):
        chosen = ('radius_fraction', 'radius_fraction_squared', 'log_radius_fraction',
                  'g_brightness_z', 'r_brightness_z', 'z_brightness_z', 'g_radial_profile_z')
        indices = np.array([names.index(n) for n in chosen])
    else:
        indices = np.arange(len(names))
    if name == 'brightness_radius_logistic' or name.startswith('full_logistic'):
        c = .1 if name.endswith('0.1') else 1.
        classifier = Pipeline([('scale', StandardScaler()),
            ('classifier', LogisticRegression(C=c, max_iter=600, random_state=SEED))])
    else:
        depth = 8 if name.endswith('8') else 12
        classifier = ExtraTreesClassifier(n_estimators=80, max_depth=depth,
            min_samples_leaf=32, max_features=.7, n_jobs=2, random_state=SEED)
    return classifier, indices


def training_samples(objects):
    xs, ys, weights, owners = [], [], [], []
    for obj in objects:
        y = obj['y3']; pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        amount = min(MAX_CLASS_PIXELS, len(pos), len(neg))
        if amount == 0:
            continue
        rng = np.random.default_rng(stable_seed(obj['annotation']['id']))
        indices = np.r_[rng.choice(pos, amount, replace=False), rng.choice(neg, amount, replace=False)]
        xs.append(obj['features'].X[indices]); ys.append(y[indices])
        # Equal total galaxy weight despite small positive masks.
        weights.append(np.full(2 * amount, 2 * MAX_CLASS_PIXELS / (2 * amount)))
        owners.extend([obj['annotation']['id']] * (2 * amount))
    if not xs:
        raise ValueError('No train galaxies have both annotation classes')
    return np.vstack(xs), np.concatenate(ys), np.concatenate(weights), np.array(owners)


def fit_candidate(name, objects, names):
    estimator, indices = candidate(name, names)
    X, y, weights, owners = training_samples(objects)
    if isinstance(estimator, Pipeline):
        estimator.fit(X[:, indices], y, scale__sample_weight=weights, classifier__sample_weight=weights)
    else:
        estimator.fit(X[:, indices], y, sample_weight=weights)
    return estimator, indices


def predict(estimator, indices, obj, full_image=False):
    features = obj['image_features'] if full_image else obj['features']
    return estimator.predict_proba(features.X[:, indices])[:, 1]


def ap(y, prediction):
    return float(average_precision_score(y, prediction)) if 0 < y.sum() < len(y) else None


def binary_metrics(y, prediction, threshold):
    binary = prediction >= threshold
    tp, fp, fn = int((binary & y).sum()), int((binary & ~y).sum()), int((~binary & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else None
    return {'precision': precision, 'recall': recall,
        'IoU': tp / (tp + fp + fn) if tp + fp + fn else None,
        'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        'candidate_pixel_fraction': float(binary.mean()),
        'Brier': float(np.mean((prediction - y) ** 2)),
        'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def probability_calibrator(estimator, indices, calibration):
    scores, labels, weights = [], [], []
    for obj in calibration:
        raw = predict(estimator, indices, obj)
        rng = np.random.default_rng(stable_seed(obj['annotation']['id']) + 1)
        sampled = rng.choice(len(raw), min(CALIBRATION_PIXELS_PER_GALAXY, len(raw)), replace=False)
        logits = np.log(np.clip(raw[sampled], 1e-6, 1 - 1e-6) / np.clip(1 - raw[sampled], 1e-6, 1))
        scores.append(logits); labels.append(obj['y3'][sampled])
        weights.append(np.full(len(sampled), CALIBRATION_PIXELS_PER_GALAXY / len(sampled)))
    classifier = LogisticRegression(C=1000, max_iter=500, random_state=SEED)
    classifier.fit(np.concatenate(scores)[:, None], np.concatenate(labels), sample_weight=np.concatenate(weights))
    # Fixed guard preserves ranking. Negative calibration slope indicates a
    # failed model/calibration relation; retain identity rather than reverse it.
    return classifier if classifier.coef_[0, 0] > 0 else None


def score_row(obj, probability, raw, baseline_prediction, source):
    y = obj['y3']
    result = {'id': obj['annotation']['id'], 'split': obj['annotation']['split'],
        'sky_group': obj['annotation']['sky_group'], 'prediction_source': source,
        'valid_pixels': len(y), 'positive_fraction': float(y.mean()),
        'model_AP': ap(y, raw), 'calibrated_model_AP': ap(y, probability),
        'uncalibrated_Brier': float(np.mean((raw-y)**2)),
        'raw_g_AP': ap(y, obj['raw_g']), 'circular_profile_AP': ap(y, obj['radial_g']),
        'brightness_radius_baseline_AP': ap(y, baseline_prediction),
        'label5_model_AP': ap(obj['y5'], raw), 'label5_raw_g_AP': ap(obj['y5'], obj['raw_g'])}
    return result


def export_prediction(obj, probability, artifact, source):
    feature = obj['image_features']; valid = feature.valid
    fields = np.full(valid.shape, np.nan, dtype=np.float32); fields[valid] = probability
    binary = np.zeros(valid.shape, np.uint8); binary[valid] = probability >= artifact['threshold']
    header = obj['images']['r'].wcs.to_header()
    header['BUNIT'] = 'annotation agreement'
    header['THRESH'] = artifact['threshold']
    header['MODEL'] = artifact['selected_model']
    header['HISTORY'] = 'Human arm-region agreement, not physical arm certainty or distance'
    header['HISTORY'] = source
    prediction_dir = OUT / 'predictions'; prediction_dir.mkdir(exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(fields, header),
        fits.ImageHDU(binary, name='CANDIDATE_MASK'),
        fits.ImageHDU(valid.astype(np.uint8), name='VALID_MASK')]).writeto(
            prediction_dir / f"{obj['annotation']['id']}_arm_regions.fits", overwrite=True)
    return {**obj['features'].descriptors,
        **probability_descriptors(probability, feature, artifact['threshold'])}


def sky_group_interval(rows, baseline_key):
    paired = [r for r in rows if r['model_AP'] is not None and r[baseline_key] is not None]
    groups = sorted({r['sky_group'] for r in paired})
    if len(groups) < 2:
        return {'sky_groups': len(groups), 'galaxies': len(paired), 'bootstrap_95_interval': None,
                'resampling_unit': 'sky_group', 'estimand': 'equal sky-group mean AP difference',
                'mean_difference': mean_or_none([r['model_AP']-r[baseline_key] for r in paired])}
    values = np.array([np.mean([r['model_AP']-r[baseline_key] for r in paired if r['sky_group']==g]) for g in groups])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {'sky_groups': len(groups), 'galaxies': len(paired),
        'resampling_unit': 'sky_group', 'estimand': 'equal sky-group mean AP difference',
        'mean_difference': float(values.mean()),
        'bootstrap_95_interval': np.quantile(draws,[.025,.975]).tolist()}


def synthetic_controls(artifact):
    """Prespecified out-of-domain arm-free controls, never used for selection."""
    from experiments.filter_synthetic_experiment import make_case
    from galaxy_lab.segmentation import predict_arm_regions
    rows = []
    for kind in ('smooth', 'bar', 'ring', 'foreground_star'):
        for index in range(4):
            images = {}
            for band, source_band in (('g','g'), ('r','r'), ('z','i')):
                image, _, geometry, _ = make_case(kind, index, source_band)
                image.band = band
                images[band] = image
            result = predict_arm_regions(artifact, images, geometry)
            rows.append({'injected_kind': kind, 'geometry_index': index,
                'valid_pixels': int(result['valid'].sum()),
                'mean_annotation_agreement': result['descriptors']['arm_probability_mean'],
                'fraction_called_candidate': result['descriptors']['arm_candidate_area_fraction'],
                'status': 'Out-of-domain analytic stress control, not observational accuracy validation'})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'frozen.json').exists():
        raise ValueError('Frozen segmentation experiment exists; preserve the one-test run. Use inference for additional images.')
    annotations_path, native_path = DATA / 'annotations/manifest.json', DATA / 'native_images/manifest.json'
    annotations_data, native_data = json.loads(annotations_path.read_text()), json.loads(native_path.read_text())
    if annotations_data.get('pending') or native_data.get('pending'):
        raise ValueError('Acquisition has pending objects: complete it before freezing this experiment')
    annotations = annotations_data['targets']
    native = {r['id']: r for r in native_data['targets']}
    group_splits = {}
    for row in annotations:
        previous = group_splits.setdefault(row['sky_group'], row['split'])
        if previous != row['split']:
            raise ValueError('A sky group crosses train/calibration/test boundaries')
    protocol = {'feature_version': FEATURE_VERSION, 'source_code_sha256': sha(__file__),
        'software_versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scikit_learn': version('scikit-learn'), 'scipy': version('scipy')},
        'feature_code_sha256': sha(ROOT / 'galaxy_lab/segmentation.py'),
        'selection_sha256': sha(DATA / 'selected_targets.json'),
        'annotation_manifest_sha256': sha(annotations_path), 'native_manifest_sha256': sha(native_path),
        'target_counts': {split: sum(r['split'] == split for r in annotations) for split in ('train','calibration','test')},
        'native_plane_selection': 'Explicit audited per-band image_hdu/invvar_hdu from a common target brick; BAND/type/shape/WCS validated, no last-HDU heuristic',
        'input_psf_handling': 'Each target matched to worst local circular Gaussian moment-equivalent PSF across g/r/z; non-Gaussian wings/covariance incomplete',
        'external_inputs': 'Human center + foreground star mask; no human arm labels in features',
        'distance_descriptor_region': 'Full native image-only valid annulus; no annotation arm labels OR annotation image footprint coverage enter descriptor pooling',
        'fixed_annulus_arcsec': [3,23], 'scales_sigma_arcsec': SCALES,
        'candidate_bank': CANDIDATES,
        'capacity_control': 'Brightness/radius feature subset uses logistic and the SAME ExtraTrees depths8/12 as full-feature models; select strongest brightness baseline by TRAIN CV only', 'candidate_selection': '3 GroupKFold splits on TRAIN sky groups only; highest mean per-galaxy AP, ties alphabetical',
        'training_pixels': 'At most400 positive and400 unlabeled per train galaxy, equal total weight per galaxy; no pixel leakage across galaxies',
        'calibration': 'Positive-slope Platt sigmoid on natural-prevalence uniform pixels; equal total galaxy weight; calibration galaxies only',
        'binary_threshold': 'Maximize calibration macro F1 across fixed grid0.05:0.05:0.95; ties choose first/smallest threshold',
        'test_policy': 'Load/test labels only after frozen choice+artifact written; no test candidate comparison or retuning',
        'uncertainty': '10000 paired whole-galaxy bootstrap for galaxy-macro differences; separate equal sky-group mean differences resampled across sky groups. The group estimand can differ from galaxy-macro table subtraction. No pixel resampling.',
        'fixed_quality_strata': 'Any independent human star-mask pixels versus none; r-band background noise above versus below TRAIN median. No threshold/model changes based on these strata.',
        'synthetic_stress': 'Four fixed analytic no-arm families smooth/bar/ring/foreground-star x4 geometries after freezing; out-of-domain illustrations, not observational accuracy or a threshold selector',
        'synthetic_generator_sha256': sha(ROOT / 'experiments/filter_synthetic_experiment.py'),
        'train_prediction_descriptors': 'Out-of-fold RAW model scores; selection of candidate used all train CV results (not nested CV). Main distance descriptors NEVER use calibration-arm labels or sigmoid; threshold fixed0.5.',
        'distance_feature_model': 'Separate artifact distance_feature_arm_model.joblib: selected TRAIN-only estimator, calibrator=None, fixedthreshold0.5 for all splits',
        'display_calibration_separation': 'arm_model.joblib and calibrated_display_descriptors.csv use separate calibration arm labels for display/segmentation only; never downstream redshift features',
        'limitations': ['Unlabeled pixels are not guaranteed non-arm negatives; outputs predict this human annotation protocol.',
            'Strong resolved spiral selection is not survey representative; no verified arm-free negative test set.',
            'Compact local features lack global segment continuity and logarithmic-arc fitting.',
            'No absolute distance, physical spacing, rotation velocity or fluid parameters enter features.',
            'This is an auditable compact baseline, not a novel or state-of-the-art segmentation claim. ZooBot:3D already published deep segmentation in2026.',
            'Pixel probabilities do not represent physically verified arm certainty.']}
    dump(OUT / 'protocol.json', protocol)
    objects, failures = {}, []
    # Test arm labels are not opened at this stage.
    for split in ('train', 'calibration'):
        for a in [r for r in annotations if r['split'] == split]:
            try:
                if a.get('status') != 'available' or native[a['id']].get('status') not in ('available','partial_coverage'):
                    raise ValueError('Acquisition failed or native science bands unavailable')
                objects[a['id']] = load_object(a, native[a['id']])
                print('FEATURES', a['id'], split, len(objects[a['id']]['y3']), flush=True)
            except Exception as exc:
                failures.append({'id': a['id'], 'split': split, 'reason': f'{type(exc).__name__}: {exc}'})
    train = [o for o in objects.values() if o['annotation']['split'] == 'train']
    calibration = [o for o in objects.values() if o['annotation']['split'] == 'calibration']
    if len(train) < 15 or len(calibration) < 5:
        raise ValueError('Too few independent galaxies for the prespecified grouped model')
    names = train[0]['features'].feature_names
    if any(o['features'].feature_names != names for o in objects.values()):
        raise ValueError('Feature schema changed between galaxies')
    folds = list(GroupKFold(3).split(np.arange(len(train)), groups=[o['annotation']['sky_group'] for o in train]))
    cv_rows, oof_by_candidate = [], {}
    for name in CANDIDATES:
        oof = {}
        for fold, (itr, iva) in enumerate(folds):
            fitting, validating = [train[k] for k in itr], [train[k] for k in iva]
            model, indices = fit_candidate(name, fitting, names)
            for obj in validating:
                full_prediction = predict(model, indices, obj, full_image=True)
                prediction = full_prediction[obj['scoring_selector']]
                identifier = obj['annotation']['id']
                oof[identifier] = full_prediction
                cv_rows.append({'candidate': name, 'fold': fold, 'id': identifier,
                    'sky_group': obj['annotation']['sky_group'], 'AP': ap(obj['y3'], prediction)})
            print('CV', name, fold, flush=True)
        oof_by_candidate[name] = oof
    save_csv(OUT / 'train_group_cv.csv', cv_rows)
    cv_means = {name: float(np.mean([r['AP'] for r in cv_rows if r['candidate'] == name and r['AP'] is not None])) for name in CANDIDATES}
    selected = sorted(CANDIDATES, key=lambda n: (-cv_means[n], n))[0]
    brightness_baseline = sorted([name for name in CANDIDATES if name.startswith('brightness_radius')], key=lambda n: (-cv_means[n],n))[0]
    estimator, indices = fit_candidate(selected, train, names)
    calibrator = probability_calibrator(estimator, indices, calibration)
    artifact = {'feature_version': FEATURE_VERSION, 'software_versions': protocol['software_versions'], 'feature_names': names,
        'feature_indices': indices, 'estimator': estimator, 'calibrator': calibrator,
        'selected_model': selected, 'scales_sigma_arcsec': SCALES,
        'required_bands': ['g','r','z'], 'threshold': .5,
        'preprocessing': 'local measured Gaussian moment-equivalent PSF matching across g/r/z; fixed common mask guards',
        'probability_meaning': 'Sigmoid-calibrated agreement with the human annotation convention, conditional on domain',
        'limitations': protocol['limitations'], 'train_ids': [o['annotation']['id'] for o in train],
        'calibration_ids': [o['annotation']['id'] for o in calibration]}
    calibration_predictions = {o['annotation']['id']: predict_feature_probabilities(artifact, o['features']) for o in calibration}
    threshold_scores = []
    for threshold in np.arange(.05, .951, .05):
        scores = [binary_metrics(o['y3'], calibration_predictions[o['annotation']['id']][0], threshold)['F1'] for o in calibration if 0 < o['y3'].sum() < len(o['y3'])]
        threshold_scores.append({'threshold': float(threshold), 'calibration_macro_F1': float(np.mean(scores))})
    artifact['threshold'] = sorted(threshold_scores, key=lambda r: (-r['calibration_macro_F1'], r['threshold']))[0]['threshold']
    baseline_model, baseline_indices = fit_candidate(brightness_baseline, train, names)
    dump(OUT / 'calibration_thresholds.json', threshold_scores)
    joblib.dump(artifact, OUT / 'arm_model.joblib', compress=3)
    distance_artifact = {**artifact, 'calibrator': None, 'threshold': .5,
        'calibration_ids': [], 'downstream_feature_model': True,
        'probability_meaning': 'Uncalibrated balanced-class classifier score; not a physical arm probability',
        'uses_calibration_arm_labels': False}
    joblib.dump(distance_artifact, OUT / 'distance_feature_arm_model.joblib', compress=3)
    frozen = {'selected_model': selected, 'train_cv_macro_AP': cv_means,
        'brightness_radius_baseline_selected_train_cv': brightness_baseline,
        'feature_names': names, 'feature_version': FEATURE_VERSION,
        'threshold': artifact['threshold'], 'calibration_sigmoid_positive_slope': calibrator is not None,
        'calibration_coef': calibrator.coef_.tolist() if calibrator is not None else None,
        'calibration_intercept': calibrator.intercept_.tolist() if calibrator is not None else None,
        'model_sha256': sha(OUT / 'arm_model.joblib'),
        'distance_feature_model_sha256': sha(OUT / 'distance_feature_arm_model.joblib'),
        'distance_features_use_calibration_arm_labels': False,
        'distance_feature_threshold_fixed': .5, 'protocol_sha256': sha(OUT / 'protocol.json'),
        'test_arm_labels_loaded_during_selection': False,
        'train_ids': artifact['train_ids'], 'calibration_ids': artifact['calibration_ids'],
        'test_ids_requested': [r['id'] for r in annotations if r['split'] == 'test'],
        'failed_before_test': failures}
    dump(OUT / 'frozen.json', frozen)
    print('FROZEN', selected, artifact['threshold'], flush=True)
    # The only main test pass, after the artifact and choice are frozen.
    for a in [r for r in annotations if r['split'] == 'test']:
        try:
            if a.get('status') != 'available' or native[a['id']].get('status') not in ('available','partial_coverage'):
                raise ValueError('Acquisition failed or native science bands unavailable')
            objects[a['id']] = load_object(a, native[a['id']])
        except Exception as exc:
            failures.append({'id': a['id'], 'split': 'test', 'reason': f'{type(exc).__name__}: {exc}'})
    rows, descriptor_rows, display_descriptor_rows, probability_cache = [], [], [], {}
    for obj in objects.values():
        a = obj['annotation']; identifier = a['id']; split = a['split']
        if split == 'train':
            raw_full = oof_by_candidate[selected][identifier]
            raw = raw_full[obj['scoring_selector']]
            if calibrator is not None:
                logits = np.log(np.clip(raw,1e-6,1-1e-6) / np.clip(1-raw,1e-6,1))
                probability = calibrator.predict_proba(logits[:,None])[:,1]
                logits_full = np.log(np.clip(raw_full,1e-6,1-1e-6) / np.clip(1-raw_full,1e-6,1))
                probability_full = calibrator.predict_proba(logits_full[:,None])[:,1]
            else:
                probability = raw
                probability_full = raw_full
            baseline_prediction = oof_by_candidate[brightness_baseline][identifier][obj['scoring_selector']]
            source = 'out_of_fold_train'
        else:
            probability_full, raw_full = predict_feature_probabilities(artifact, obj['image_features'])
            probability, raw = probability_full[obj['scoring_selector']], raw_full[obj['scoring_selector']]
            baseline_prediction = predict(baseline_model, baseline_indices, obj)
            source = 'train_only_model_calibration' if split == 'calibration' else 'frozen_model_test'
        score = score_row(obj, probability, raw, baseline_prediction, source)
        score.update(binary_metrics(obj['y3'], probability, artifact['threshold']))
        # No well-defined AP means no complete positive/negative spatial truth
        # in this fixed annulus; report instead of treating it as an arm-free test.
        score['annotation_classes_status'] = 'both_classes' if score['model_AP'] is not None else 'insufficient_two_class_labels'
        rows.append(score)
        display_desc = export_prediction(obj, probability_full, artifact, source)
        raw_desc = {**obj['image_features'].descriptors, **probability_descriptors(raw_full, obj['image_features'], .5)}
        display_descriptor_rows.append({'id': identifier, 'split': split, 'sky_group': a['sky_group'],
            'prediction_source': source, 'feature_use': 'calibrated display only; do not use for redshift calibration', **display_desc})
        descriptor_rows.append({'id': identifier, 'split': split, 'sky_group': a['sky_group'],
            'prediction_source': source, 'arm_score_kind': 'raw uncalibrated balanced-class classifier score',
            'arm_feature_threshold': .5, 'arm_feature_model': 'distance_feature_arm_model.joblib',
            'uses_calibration_arm_labels': False, **raw_desc,
            'valid_pixels': obj['metadata']['image_feature_valid_pixels'],
            'geometric_pixels': obj['metadata']['image_geometric_pixels'],
            'coverage_fraction': obj['metadata']['image_feature_valid_pixels'] / obj['metadata']['image_geometric_pixels']})
        probability_cache[identifier] = probability_full
    save_csv(OUT / 'galaxy_scores.csv', rows)
    save_csv(OUT / 'image_descriptors.csv', descriptor_rows)
    save_csv(OUT / 'calibrated_display_descriptors.csv', display_descriptor_rows)
    dump(OUT / 'coverage.json', [o['metadata'] for o in objects.values()])
    dump(OUT / 'failures.json', failures)
    test_rows = [r for r in rows if r['split'] == 'test' and r['model_AP'] is not None]
    def paired(key):
        return paired_interval([r['model_AP'] - r[key] for r in test_rows if r[key] is not None])
    summary = {'frozen_choice': frozen, 'test_requested': sum(a['split']=='test' for a in annotations),
        'test_predictions_completed': sum(r['split']=='test' for r in rows),
        'test_two_class_AP_defined': len(test_rows), 'failed': failures,
        'test_macro_AP': mean_or_none([r['model_AP'] for r in test_rows]),
        'test_macro_calibrated_AP': mean_or_none([r['calibrated_model_AP'] for r in test_rows]),
        'test_macro_uncalibrated_Brier': mean_or_none([r['uncalibrated_Brier'] for r in test_rows]),
        'test_macro_raw_g_AP': mean_or_none([r['raw_g_AP'] for r in test_rows]),
        'test_macro_circular_profile_AP': mean_or_none([r['circular_profile_AP'] for r in test_rows]),
        'test_macro_brightness_radius_baseline_AP': mean_or_none([r['brightness_radius_baseline_AP'] for r in test_rows]),
        'model_minus_raw_g_AP': paired('raw_g_AP'),
        'model_minus_circular_profile_AP': paired('circular_profile_AP'),
        'model_minus_brightness_radius_baseline_AP': paired('brightness_radius_baseline_AP'),
        'test_sky_groups': len({r['sky_group'] for r in test_rows}),
        'sky_group_cluster_bootstrap': {key: sky_group_interval(test_rows,key) for key in ('raw_g_AP','circular_profile_AP','brightness_radius_baseline_AP')},
        'test_macro_binary': {key: mean_or_none([r[key] for r in test_rows]) for key in ('precision','recall','IoU','F1','Brier')},
        'label5_sensitivity': {'defined_galaxies': sum(r['label5_model_AP'] is not None for r in test_rows),
            'macro_model_AP': mean_or_none([r['label5_model_AP'] for r in test_rows]),
            'macro_raw_g_AP': mean_or_none([r['label5_raw_g_AP'] for r in test_rows]),
            'paired_difference': paired_interval([r['label5_model_AP'] - r['label5_raw_g_AP'] for r in test_rows if r['label5_model_AP'] is not None and r['label5_raw_g_AP'] is not None])},
        'descriptor_rows': len(descriptor_rows),
        'inference_artifact': 'arm_model.joblib',
        'downstream_image_descriptors_artifact': 'distance_feature_arm_model.joblib',
        'downstream_image_descriptors_use_calibration_arm_labels': False,
        'downstream_threshold': .5,
        'interpretation': 'Agreement with incomplete human arm polygons in a selected resolved-spiral cohort; no physical arm-distance or fluid-microphysics validation.',
        'posthoc_analysis': False}
    # Fixed quality strata diagnostics use labels only after freezing; they are
    # descriptive, not a new selector or a basis for threshold retuning.
    lookup = {o['annotation']['id']: o for o in objects.values()}
    train_noise = [o['features'].descriptors['r_background_noise'] for o in train]
    noise_cut = float(np.median(train_noise))
    quality = {}
    for label, predicate in {
        'human_star_pixels_present': lambda o: o['metadata']['star_mask_pixels'] > 0,
        'no_human_star_pixels': lambda o: o['metadata']['star_mask_pixels'] == 0,
        'higher_noise_than_train_median': lambda o: o['features'].descriptors['r_background_noise'] > noise_cut,
        'lower_noise_than_train_median': lambda o: o['features'].descriptors['r_background_noise'] <= noise_cut}.items():
        subset = [r for r in test_rows if predicate(lookup[r['id']])]
        quality[label] = {'galaxies': len(subset), 'macro_AP': float(np.mean([r['model_AP'] for r in subset])) if subset else None}
    summary['fixed_quality_strata'] = quality
    stress_rows = synthetic_controls(artifact)
    save_csv(OUT / 'synthetic_stress_controls.csv', stress_rows)
    summary['out_of_domain_synthetic_controls'] = {kind: {
        'images': sum(r['injected_kind'] == kind for r in stress_rows),
        'mean_false_candidate_fraction': float(np.mean([r['fraction_called_candidate'] for r in stress_rows if r['injected_kind'] == kind]))}
        for kind in ('smooth','bar','ring','foreground_star')}
    dump(OUT / 'summary.json', summary)
    (OUT / 'README.txt').write_text(
        'Supervised human arm-region prototype\n=====================================\n\n'
        + f"Selected from TRAIN-only grouped CV: {selected}.\n"
        + f"Held-out test requested/scored with two-class AP: {summary['test_requested']}/{summary['test_two_class_AP_defined']}.\n"
        + f"Macro AP model/raw g/circular-profile/brightness-radius baseline: {summary['test_macro_AP']}/{summary['test_macro_raw_g_AP']}/{summary['test_macro_circular_profile_AP']}/{summary['test_macro_brightness_radius_baseline_AP']}.\n\n"
        + 'arm_model.joblib is the segmentation display model: train-only estimator plus sigmoid/threshold fitted on separate calibration arm annotations.\n'
        + 'distance_feature_arm_model.joblib shares the train-only estimator, with NO sigmoid and fixed threshold0.5. It NEVER uses calibration arm labels.\n'
        + 'image_descriptors.csv contains RAW distance-model scores: OOF predictions for TRAIN galaxies, frozen train-only predictions for CALIBRATION/TEST.\n'
        + 'Despite historical arm_probability_* column names, these raw scores are not calibrated probabilities. All pooling uses an image-only native annulus, independent of human arm labels or annotation footprint coverage.\n'
        + 'calibrated_display_descriptors.csv and predictions/*.fits are for segmentation display/validation only, not redshift calibration.\n\n'
        + 'All inputs are native, PSF approximately homogenized g/r/z, supplied human centers and independent foreground-star masks. The fixed sky annulus is3-23arcsec.\n'
        + 'Unlabeled pixels are uncertain negatives. The strong-spiral cohort supplies no verified arm-free observational test population.\n'
        + 'Pixel masks estimate agreement with GZ3D annotation conventions; they do not establish arm segments, pitch, physical spacing, distance or viscosity.\n'
        + 'See protocol/frozen/summary JSONs, full galaxy scores, coverage/failures, paired whole-galaxy and sky-group bootstrap intervals and five-vote sensitivity.\n'
        + 'Synthetic stress controls are out-of-domain diagnostic illustrations, never selection/calibration data or observational accuracy evidence.\n\n'
        + 'Reproduce into a NEW empty output location: OPENBLAS_NUM_THREADS=1 python experiments/supervised_arm_experiment.py --output-dir /tmp/galaxy_segmentation_reproduction\n'
        + 'Reusing this known test set is computational reproduction, not new confirmatory evidence. Existing frozen experiments are preserved.\n')
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    axes[0].barh(list(cv_means), list(cv_means.values())); axes[0].set(xlabel='Train-group CV macro AP',title='Model choice used train galaxies only')
    axes[1].scatter([r['raw_g_AP'] for r in test_rows], [r['model_AP'] for r in test_rows]); axes[1].plot([0,1],[0,1],color='gray')
    axes[1].set(xlabel='Raw g brightness AP',ylabel='Frozen model AP',title=f'Unseen test galaxies (n={len(test_rows)})',xlim=(0,1),ylim=(0,1))
    fig.savefig(OUT / 'model_comparison.png',dpi=150);plt.close(fig)
    examples = sorted([o for o in objects.values() if o['annotation']['split']=='test'],key=lambda o:o['annotation']['rank'])[:6]
    if not examples:
        print(json.dumps(safe_json(summary),indent=2),flush=True)
        return
    fig, axes = plt.subplots(len(examples),4,figsize=(12,3*len(examples)),constrained_layout=True,squeeze=False)
    for row,obj in enumerate(examples):
        identifier=obj['annotation']['id'];valid=obj['image_features'].valid
        field=np.full(valid.shape,np.nan);field[valid]=probability_cache[identifier]
        axes[row,0].imshow(display_transform(obj['images']['r'].data),origin='lower',cmap='gray');axes[row,0].set_title(identifier+' native r')
        axes[row,1].imshow(np.where(obj['features'].valid,obj['votes']>=3,np.nan),origin='lower',cmap='gray',vmin=0,vmax=1);axes[row,1].set_title('Independent human votes≥3')
        axes[row,2].imshow(field,origin='lower',cmap='magma',vmin=0,vmax=1);axes[row,2].set_title('Frozen annotation-agreement model')
        axes[row,3].imshow(np.where(valid,field>=artifact['threshold'],np.nan),origin='lower',cmap='gray',vmin=0,vmax=1);axes[row,3].set_title(f"One calibration threshold={artifact['threshold']:.2f}")
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('First six held-out IDs in frozen order; no best-example selection')
    fig.savefig(OUT / 'heldout_predictions.png',dpi=140);plt.close(fig)
    print(json.dumps(safe_json(summary),indent=2),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='New experiment output location; existing frozen results are preserved')
    args = parser.parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir
    main()
