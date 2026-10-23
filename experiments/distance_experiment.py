"""Execute the public SDSS catalog-size/shape redshift pilot (not arm features)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR","/tmp/galaxy_lab_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/galaxy_lab_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from galaxy_lab.distance import evaluate_distance_features, _json_safe


def write_subgroup_audit(predictions,data,out):
    """Diagnostic slices only; no fitting or selection from these outcomes."""
    labeled=predictions.merge(data[["galaxy_id","angular_r50_arcsec","axis_ratio"]],
                              on="galaxy_id",how="left",validate="one_to_one")
    subgroup_sets={
        "spec_z_0.02_to_0.05":(labeled.spec_z>=.02)&(labeled.spec_z<.05),
        "spec_z_0.05_to_0.10":(labeled.spec_z>=.05)&(labeled.spec_z<.1),
        "spec_z_0.10_to_0.20":(labeled.spec_z>=.1)&(labeled.spec_z<=.2),
        "angular_r50_lt4_arcsec":labeled.angular_r50_arcsec<4,
        "angular_r50_ge4_arcsec":labeled.angular_r50_arcsec>=4,
        "axis_ratio_lt0.5":labeled.axis_ratio<.5,
        "axis_ratio_ge0.5":labeled.axis_ratio>=.5,
    }
    rows=[]
    for subgroup,selected in subgroup_sets.items():
        subset=labeled.loc[selected]
        for model in ["median","photometry","photometry_morphology"]:
            pred=subset[f"{model}_prediction"]
            error=pred-subset.spec_z
            covered=(subset.spec_z>=subset[f"{model}_lower"])&(subset.spec_z<=subset[f"{model}_upper"])
            rows.append({"subgroup":subgroup,"model":model,"galaxies":len(subset),
                         "sky_groups":subset.group_id.nunique(),
                         "mae_redshift":float(error.abs().mean()) if len(subset) else np.nan,
                         "mean_signed_redshift_error":float(error.mean()) if len(subset) else np.nan,
                         "mean_signed_delta_z_over_1plusz":float((error/(1+subset.spec_z)).mean()) if len(subset) else np.nan,
                         "nominal90_galaxy_interval_coverage":float(covered.mean()) if len(subset) else np.nan})
    pd.DataFrame(rows).to_csv(Path(out)/"subgroup_audit.csv",index=False)
    (Path(out)/"subgroup_audit_notes.txt").write_text(
        "Diagnostic held-out slices: spectroscopic z .02-.05,.05-.10,.10-.20;\n"
        "angular r50 below/above4 arcsec; axis ratio below/above.5.\n"
        "No refitting, retuning or feature selection uses these outcomes.\n"
        "Denominators are reported; small groups have unstable errors/coverage.\n"
        "Conformal intervals are marginal over comparable groups, not guaranteed\n"
        "within any subgroup. These slices do not establish population-wide\n"
        "or shifted-survey calibration. Label-selection limits remain.\n")


def run(source=ROOT/"executed_research/images/sdss_resolved_sample.csv",
        out=ROOT/"executed_research/distance_observed"):
    source,out = Path(source),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    raw = source.read_bytes()
    data = pd.read_csv(source,comment="#",dtype={"galaxy_id":str,"specObjID":str})
    # Coordinates organize the split only; they never become predictors.
    data["group_id"] = (np.floor(data.ra/10).astype(int).astype(str)+"_"+
                        np.floor((data.dec+90)/10).astype(int).astype(str))
    for first,second in [("u","g"),("g","r"),("r","i"),("i","z")]:
        data[f"color_{first}_{second}"] = data[f"mag_{first}"]-data[f"mag_{second}"]
    photo = [f"mag_{band}" for band in "ugriz"] + ["color_u_g","color_g_r","color_r_i","color_i_z"]
    morph = ["angular_r50_arcsec","concentration","axis_ratio"]
    data.to_csv(out/"sdss_labeled_features.csv",index=False)
    provenance = {
        "input_csv":str(source.resolve()),"raw_sha256":hashlib.sha256(raw).hexdigest(),
        "source":"official SDSS DR17 SkyServer public catalog query",
        "label":"independent spectroscopic redshift, spec_z",
        "photo":"ugriz modelMag-extinction, with adjacent colors derived from those apparent magnitudes",
        "morphology":"r-band angular half-light radius, concentration and axis ratio; not spiral-arm measurements",
        "selection":"deterministic TOP500 clean resolved galaxies, .02<spec_z<.2, 3<r50<12 arcsec, valid ugriz/axisratio",
        "selection_order":"CHECKSUM(specObjID,20261023); exact acquisition SQL archived by acquisition workflow",
        "groups":"10 degree coordinate tiles: floor(ra/10),floor((dec+90)/10)",
        "group_caveat":"rectangular sky blocks are a survey leakage precaution, not a guarantee of physically independent galaxies",
    }
    acquisition_record = source.parent/"sdss_query_provenance.json"
    if acquisition_record.exists():
        provenance["acquisition_record"] = str(acquisition_record.resolve())
        provenance["acquisition_record_sha256"] = hashlib.sha256(acquisition_record.read_bytes()).hexdigest()
        provenance["acquisition_query"] = json.loads(acquisition_record.read_text())["query"]
    summary,predictions,manifest = evaluate_distance_features(
        data,photo,morph,target="spec_z",provenance=provenance,
        seed=20261023,alpha=.1,n_estimators=128,n_bootstrap=4000)
    # All coverage levels reuse identical fixed features, split, tree seed and
    # hyperparameters. No choice is made based on test performance.
    summary["additional_nominal_interval_coverages"] = {}
    for alpha in [.32,.05]:
        alternative,_,alternative_manifest = evaluate_distance_features(
            data,photo,morph,target="spec_z",provenance=provenance,
            seed=20261023,alpha=alpha,n_estimators=128,n_bootstrap=100)
        if not manifest.equals(alternative_manifest):
            raise RuntimeError("coverage-level runs must use identical splits")
        summary["additional_nominal_interval_coverages"][f"{1-alpha:.2f}"] = {
            name:{key:record[key] for key in ["galaxy_interval_coverage","whole_group_interval_coverage",
                                             "median_interval_width_native","conformal_radius_transformed"]}
            for name,record in alternative["models"].items()}
    (out/"distance_summary.json").write_text(json.dumps(_json_safe(summary),indent=2,allow_nan=False)+"\n")
    predictions.to_csv(out/"distance_test_predictions.csv",index=False)
    manifest.to_csv(out/"distance_split_manifest.csv",index=False)
    write_subgroup_audit(predictions,data,out)
    fig,axes = plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    truth = predictions.spec_z.to_numpy()
    limits = (min(.015,float(truth.min())*.9), max(.205,float(truth.max())*1.05))
    for ax,name,title in zip(axes,["photometry","photometry_morphology"],
                             ["Apparent magnitudes + colors","Same photometry + catalog size / shape"]):
        pred = predictions[f"{name}_prediction"].to_numpy()
        ax.scatter(truth,pred,s=20,alpha=.7,color="#245c80")
        ax.plot(limits,limits,"--",color="0.5",linewidth=1)
        ax.set(xlabel="Held-out spectroscopic redshift",ylabel="Predicted redshift",
               title=title,xlim=limits,ylim=limits)
        ax.text(.04,.96,f"MAE = {summary['models'][name]['mae_native_target_units']:.4f}",
                transform=ax.transAxes,va="top")
    fig.suptitle("Public SDSS pilot: selected resolved galaxies; sky-group holdout")
    fig.savefig(out/"redshift_test_predictions.png",dpi=170)
    plt.close(fig)
    bootstrap = summary["paired_morphology_improvement"]
    (out/"INTERPRETATION.txt").write_text(
        "PUBLIC SDSS CATALOG REDSHIFT PILOT — OBSERVATIONAL, EXPLORATORY\n\n"
        f"Input: {len(data)} distinct galaxies, {data.group_id.nunique()} ten-degree sky tiles.\n"
        f"Test: {len(predictions)} galaxies in {summary['split_groups']['test']} held-out tiles.\n"
        "Predictors: extinction-corrected ugriz apparent magnitudes plus adjacent colors;\n"
        "augmentation adds angular r50, concentration and axis ratio. No arm extraction\n"
        "or viscosity/fluid-model feature is used. Labels are spectroscopic redshift.\n"
        "No conversion from redshift to physical distance is performed.\n\n"
        +"\n".join(f"{name}: MAE(z)={rec['mae_native_target_units']:.6f}, "
                     f"NMAD(delta z/(1+z))={rec['sigma_nmad_delta_z_over_1plusz']:.6f}, "
                     f"nominal90% galaxy coverage={rec['galaxy_interval_coverage']:.3f}, "
                     f"whole-group coverage={rec['whole_group_interval_coverage']:.3f}"
                     for name,rec in summary["models"].items())
        +f"\n\nPaired morphology improvement in mean absolute ln(1+z) error: {bootstrap['mean_improvement']:.6f}; "
        f"group-bootstrap95% interval [{bootstrap['ci95_low']:.6f},{bootstrap['ci95_high']:.6f}].\n"
        "Positive means lower error after adding catalog morphology. This estimates\n"
        "the incremental association on this selected sample, not a universal ruler.\n\n"
        "Limitations: one small preselected resolved sample, one fixed split; catalog\n"
        "PSF/measurement and label uncertainties are not propagated; bootstrap holds\n"
        "the trained models fixed; sky tiles do not guarantee cosmic independence.\n"
        "This is no evidence for viscosity and no validation of spiral-arm distance\n"
        "inference. A larger independently selected external survey test, calibrated\n"
        "raw-image morphology with PSF/selection checks, and full provenance are\n"
        "required before practical use or accuracy claims beyond this pilot.\n")
    print(json.dumps(_json_safe({"rows":len(data),"split_groups":summary["split_groups"],
                                "models":summary["models"],"paired_morphology_improvement":bootstrap}),indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"executed_research/images/sdss_resolved_sample.csv")
    parser.add_argument("--out",type=Path,default=ROOT/"executed_research/distance_observed")
    arguments = parser.parse_args()
    run(arguments.source,arguments.out)
