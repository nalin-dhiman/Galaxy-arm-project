"""Predeclared small arm-descriptor ablation on the first 160 SDSS pilot rows.

The catalog was preselected in spectroscopic redshift and angular size. Within
that selected population, no further label-dependent image selection, quality
gating, feature tuning or hyperparameter selection is performed here. Image
extraction defines its quality gates upstream. All selected galaxies remain in
every model, including image/extraction failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.distance import (
    _json_safe, evaluate_distance_features, group_disjoint_split,
    paired_group_bootstrap,
)

BASELINE=([f"mag_{b}" for b in "ugriz"]+["color_u_g","color_g_r","color_r_i","color_i_z"]
          +["angular_r50_arcsec","concentration","axis_ratio"])
FLAGS=["arm_image_available","arm_candidate_accepted"]
ARM_FEATURES=["pitch_abs_deg","arm_mode","fourier_amplitude",
              "arm_radial_mode_support","arm_mode_dominance_ratio"]


def run(source=ROOT/"executed_research/images/sdss_resolved_sample.csv",
        image_features=ROOT/"executed_research/sample_images/image_features.csv",
        out=ROOT/"executed_research/distance_arms"):
    source,image_features,out=Path(source),Path(image_features),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv(source,comment="#",dtype={"galaxy_id":str,"specObjID":str}).head(160).copy()
    images=pd.read_csv(image_features,dtype={"galaxy_id":str})
    if images.galaxy_id.duplicated().any():
        raise ValueError("image feature rows must have unique galaxy IDs")
    if set(images.galaxy_id)!=set(catalog.galaxy_id):
        raise ValueError("the image manifest must preserve all and only the original first160 galaxies")
    images=images.rename(columns={"image_available":"arm_image_available"})
    cols=["galaxy_id"]+FLAGS+ARM_FEATURES
    missing=[name for name in cols if name not in images]
    if missing:
        raise ValueError(f"image feature interface is missing columns: {missing}")
    data=catalog.merge(images[cols],on="galaxy_id",how="left",validate="one_to_one",sort=False)
    if not data.galaxy_id.equals(catalog.reset_index(drop=True).galaxy_id):
        raise RuntimeError("catalog order changed during image feature crossmatch")
    data["group_id"]=(np.floor(data.ra/10).astype(int).astype(str)+"_"+
                      np.floor((data.dec+90)/10).astype(int).astype(str))
    for first,second in [("u","g"),("g","r"),("r","i"),("i","z")]:
        data[f"color_{first}_{second}"]=data[f"mag_{first}"]-data[f"mag_{second}"]
    for col in FLAGS:
        data[col]=pd.to_numeric(data[col],errors="raise").astype(int)
        if not data[col].isin([0,1]).all():
            raise ValueError(f"{col} must be an explicit binary flag")
    for col in ARM_FEATURES:
        data[col]=pd.to_numeric(data[col],errors="raise")
    rejected=data.arm_candidate_accepted==0
    if data.loc[rejected,ARM_FEATURES].notna().any().any():
        raise ValueError("rejected candidates must have all arm feature values missing")
    data.to_csv(out/"sdss_arm_labeled_features.csv",index=False)
    split=group_disjoint_split(data.group_id)
    audit=[]
    for partition in ["train","calibration","test"]:
        selected=data.loc[split==partition]
        audit.append({"split":partition,"galaxies":len(selected),
                      "groups":selected.group_id.nunique(),
                      "images_available":int(selected.arm_image_available.sum()),
                      "accepted_algorithm_candidates":int(selected.arm_candidate_accepted.sum()),
                      **{f"missing_{f}":int(selected[f].isna().sum()) for f in ARM_FEATURES}})
    pd.DataFrame(audit).to_csv(out/"missingness_by_split.csv",index=False)
    provenance={
        "catalog_source":str(source.resolve()),"catalog_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "image_features_source":str(image_features.resolve()),"image_features_sha256":hashlib.sha256(image_features.read_bytes()).hexdigest(),
        "selection":"first160 original deterministic catalog rows, no label-dependent image selection",
        "image_processing":"256x256 r-band cutouts at 0.396 arcsec/pixel; upstream predeclared candidate quality gates",
        "labels":"SDSS spectroscopic redshifts; not metric distances",
        "groups":"10degree RA/Dec coordinate tiles, same rule/seed as500 catalog pilot",
        "split_caveat":"160 and500 experiments have different group universes and thus different partitions; all models within160 use identical partitions",
        "caveat":"accepted means passing algorithm quality gates, not externally confirmed spiral arms",
        "baseline_features":BASELINE,"availability_control":FLAGS,"arm_features":ARM_FEATURES,
    }
    # A -> B isolates image availability and candidate acceptance information.
    availability,pred_avail,manifest=evaluate_distance_features(
        data,BASELINE,FLAGS,target="spec_z",n_estimators=128,n_bootstrap=4000,provenance=provenance,
        missing_indicators=False)
    if (data.loc[split=="train",ARM_FEATURES].nunique(dropna=True).max()<2
            or data.loc[split=="calibration","arm_candidate_accepted"].sum()==0
            or data.loc[split=="test","arm_candidate_accepted"].sum()==0):
        # One accepted training candidate supplies no geometric variation to
        # learn. Do not interpret effects of duplicate missingness indicators
        # or constant imputation as evidence for a pitch-distance relation.
        reason="Arm descriptors need variation in training and accepted candidates in calibration and test; geometric/pitch increment is not estimable."
        summary={"experiment_kind":"observational exploratory160galaxy arm-descriptor pilot",
                 "status":"arm_feature_increment_not_estimable","reason":reason,
                 "rows":len(data),"images_available":int(data.arm_image_available.sum()),
                 "accepted_algorithm_candidates":int(data.arm_candidate_accepted.sum()),
                 "accepted_by_split":{s:int(data.loc[split==s,"arm_candidate_accepted"].sum()) for s in ["train","calibration","test"]},
                 "split_rows":availability["split_rows"],"split_groups":availability["split_groups"],
                 "provenance":provenance,"models":{
                     "catalog_baseline":availability["models"]["photometry"],
                     "baseline_plus_availability":availability["models"]["photometry_morphology"]},
                 "paired_improvements":{"availability_vs_catalog":availability["paired_morphology_improvement"]},
                 "limits":["no externally confirmed arm labels","only availability increment could be evaluated",
                           "quality gates kept fixed; no post-result relaxation","selected160galaxy redshift-restricted pilot"]}
        (out/"summary.json").write_text(json.dumps(_json_safe(summary),indent=2,allow_nan=False)+"\n")
        (out/"availability.json").write_text(json.dumps(_json_safe(availability),indent=2,allow_nan=False)+"\n")
        pred_avail.to_csv(out/"distance_test_predictions.csv",index=False)
        manifest.to_csv(out/"distance_split_manifest.csv",index=False)
        paired=availability["paired_morphology_improvement"]
        (out/"INTERPRETATION.txt").write_text(
            "REAL-IMAGE ARM INCREMENT: NOT ESTIMABLE\n\n"+reason+"\n"
            +f"Selected160; images available={int(data.arm_image_available.sum())}; "
             f"accepted candidates={int(data.arm_candidate_accepted.sum())}.\n"
            +f"Acceptance by split: {summary['accepted_by_split']}.\n"
            +"All selected rows retained; conservative image quality gates remain fixed.\n"
            +"Baseline includes the same adjacent colors as the500galaxy experiment.\n"
            +"The160 and500 group universes differ; only within160 model comparisons\n"
            +"share test galaxies. Their headline errors must not be compared as paired.\n"
            +"Full-arm and pitch-only regressions were not fit because a lone finite\n"
            +"training candidate cannot teach a geometric relationship; zero held-out\n"
            +"accepted candidates cannot test one. No successful arm-distance claim.\n\n"
            +f"Availability-only paired improvement in absolute ln(1+z) error={paired['mean_improvement']:.6g}; "
             f"95%group-bootstrapCI=[{paired['ci95_low']:.6g},{paired['ci95_high']:.6g}].\n"
            +"This control concerns image availability/selection, not arms or viscosity.\n")
        print(json.dumps(_json_safe(summary),indent=2,allow_nan=False))
        return
    # B -> C uses explicit availability flags once. Disable automatic imputer
    # indicators so C does not duplicate acceptance flags for every descriptor.
    full,pred_full,full_manifest=evaluate_distance_features(
        data,BASELINE+FLAGS,ARM_FEATURES,target="spec_z",n_estimators=128,n_bootstrap=4000,provenance=provenance,
        missing_indicators=False)
    # Predeclared pitch-only increment avoids attributing mode/strength gains
    # specifically to pitch. This remains an exploratory multiple comparison.
    pitch,pred_pitch,pitch_manifest=evaluate_distance_features(
        data,BASELINE+FLAGS,["pitch_abs_deg"],target="spec_z",n_estimators=128,n_bootstrap=4000,provenance=provenance,
        missing_indicators=False)
    if not manifest.equals(full_manifest) or not manifest.equals(pitch_manifest):
        raise RuntimeError("all ablations must preserve identical grouped splits")
    if not pred_avail.galaxy_id.equals(pred_full.galaxy_id) or not pred_avail.galaxy_id.equals(pred_pitch.galaxy_id):
        raise RuntimeError("all ablations must predict the same test galaxies")
    np.testing.assert_array_equal(pred_avail.photometry_morphology_prediction.to_numpy(),pred_full.photometry_prediction.to_numpy())
    transformed=np.log1p(pred_avail.spec_z.to_numpy())
    baseline_error=np.abs(np.log1p(pred_avail.photometry_prediction.to_numpy())-transformed)
    full_error=np.abs(np.log1p(pred_full.photometry_morphology_prediction.to_numpy())-transformed)
    full_vs_baseline=paired_group_bootstrap(pred_avail.group_id,baseline_error,full_error,n_bootstrap=4000)
    summary={
        "experiment_kind":"observational exploratory160galaxy arm-descriptor pilot",
        "provenance":provenance,"rows":len(data),"split_rows":full["split_rows"],"split_groups":full["split_groups"],
        "models":{
            "catalog_baseline":availability["models"]["photometry"],
            "baseline_plus_availability":availability["models"]["photometry_morphology"],
            "baseline_plus_availability_and_arm_descriptors":full["models"]["photometry_morphology"],
            "baseline_plus_availability_and_pitch_only":pitch["models"]["photometry_morphology"],
        },
        "paired_improvements":{
            "availability_vs_catalog":availability["paired_morphology_improvement"],
            "full_descriptors_vs_availability":full["paired_morphology_improvement"],
            "pitch_only_vs_availability":pitch["paired_morphology_improvement"],
            "full_descriptors_vs_catalog":full_vs_baseline,
        },
        "limits":["very small redshift- and angular-size-selected sample and one split","exploratory multiple ablations; no significance correction",
                  "quality acceptance is not external arm validation","missingness can encode selection and surface brightness",
                  "sky tile groups are a leakage precaution, not proof of physical independence",
                  "no fluid/viscosity predictor or metric distance conversion"],
    }
    predictions=pred_avail[["galaxy_id","group_id","spec_z"]].copy()
    for label,predictions_source,model in [
        ("catalog_baseline",pred_avail,"photometry"),
        ("availability",pred_avail,"photometry_morphology"),
        ("full_arm_descriptors",pred_full,"photometry_morphology"),
        ("pitch_only",pred_pitch,"photometry_morphology")]:
        for suffix in ["prediction","lower","upper"]:
            predictions[f"{label}_{suffix}"]=predictions_source[f"{model}_{suffix}"]
    predictions.to_csv(out/"distance_test_predictions.csv",index=False)
    manifest.to_csv(out/"distance_split_manifest.csv",index=False)
    for name,record in [("availability",availability),("full",full),("pitch_only",pitch),("summary",summary)]:
        (out/f"{name}.json").write_text(json.dumps(_json_safe(record),indent=2,allow_nan=False)+"\n")
    lines=["EXPLORATORY REAL-IMAGE ARM-DESCRIPTOR REDSHIFT PILOT", "",
           f"Selected160 galaxies; {int(data.arm_image_available.sum())} images available; "
           f"{int(data.arm_candidate_accepted.sum())} pass algorithm candidate gates.",
           "All160 remain in every model; missing descriptors use training-only imputation.",
           "Same sky-group splits and test galaxies for all comparisons. No metric distances.",""]
    lines += [f"{name}: MAE(z)={record['mae_native_target_units']:.6f}" for name,record in summary["models"].items()]
    lines += [""]+[f"{name}: paired mean improvement={record['mean_improvement']:.6g}, "
                    f"bootstrap95%CI=[{record['ci95_low']:.6g},{record['ci95_high']:.6g}]"
                    for name,record in summary["paired_improvements"].items()]
    lines += ["", "Positive means lower absolute ln(1+z) error; intervals bootstrap sky groups.",
              "Accepted candidates have no external spiral tracing truth. Any observed gain",
              "is exploratory and may reflect quality/selection; no evidence for viscosity.",
              "A zero-crossing interval does not establish incremental predictive benefit."]
    (out/"INTERPRETATION.txt").write_text("\n".join(lines)+"\n")
    print(json.dumps(_json_safe(summary),indent=2,allow_nan=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"executed_research/images/sdss_resolved_sample.csv")
    parser.add_argument("--image-features",type=Path,default=ROOT/"executed_research/sample_images/image_features.csv")
    parser.add_argument("--out",type=Path,default=ROOT/"executed_research/distance_arms")
    args=parser.parse_args()
    run(args.source,args.image_features,args.out)
