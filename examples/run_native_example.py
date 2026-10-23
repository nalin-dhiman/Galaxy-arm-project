#!/usr/bin/env python3
"""Run the bundled calibrated-image example with its matching frozen models."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.__main__ import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    center = json.loads((ROOT / "examples/data/center.json").read_text())
    main(["analyze-guarded", "--image", str(ROOT / "examples/data/galaxy_grz.fits"),
          "--psf", str(ROOT / "examples/data/galaxy_psf.fits"), "--units", "nanomaggy/pixel",
          "--foreground-mask", str(ROOT / "examples/data/foreground_mask.fits"),
          "--center-ra", str(center["ra_deg"]), "--center-dec", str(center["dec_deg"]),
          "--arm-model", str(ROOT / "models/arms/arm_model.joblib"),
          "--distance-arm-model", str(ROOT / "models/arms/distance_feature_arm_model.joblib"),
          "--redshift-model", str(ROOT / "models/redshift/P/redshift_model.joblib"),
          "--catalog", str(ROOT / "examples/data/observed_photometry.csv"),
          "--galaxy-id", center["galaxy_id"], "--survey", "legacy-dr9",
          "--population", "strong-resolved-spiral", "--output-dir", str(args.output_dir)])
