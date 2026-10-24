"""Independent result-table checks, including original quoted-error scoring."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wide-bounds', action='store_true')
parser.add_argument('--reference-fits', type=Path,
                    help='Optional independent fit CSV for matching NFW/Burkert chi-square checks')
args = parser.parse_args()
OUT = ROOT/"executed_research/dynamics"
if args.wide_bounds:
    OUT = OUT/"wide_bounds"

full = pd.read_csv(OUT/"full_fits.csv")
held = pd.read_csv(OUT/"holdout_fits.csv")
points = pd.read_csv(OUT/"outer_predictions.csv")
splits = pd.read_csv(OUT/"galaxy_splits.csv")
assert len(full) == 700
assert len(held) == 628
assert not full.duplicated(["galaxy", "model"]).any()
assert not held.duplicated(["galaxy", "model"]).any()
assert (full.k == 2).all() and (held.k == 2).all()
assert full.success.all() and held.success.all() and held.test_prediction_valid.all()

maximum_metric_error = 0.
for row in held.itertuples(index=False):
    test = points[(points.galaxy == row.galaxy) & (points.model == row.model) & (points.partition == "test")]
    train = points[(points.galaxy == row.galaxy) & (points.model == row.model) & (points.partition == "train")]
    assert len(test) == row.n_test >= 2
    assert len(train) == row.n >= 5
    assert test.radius_kpc.min() > train.radius_kpc.max()
    rmse = float(np.sqrt(np.mean((test.v_predicted-test.v_observed)**2)))
    normalized = float(np.mean(((test.v_predicted-test.v_observed)/test.error_kms)**2))
    assert np.isclose(rmse, row.test_rmse, rtol=1e-12, atol=1e-10)
    assert np.isclose(normalized, row.test_chi2_per_point, rtol=1e-12, atol=1e-10)
    maximum_metric_error = max(maximum_metric_error, abs(rmse-row.test_rmse), abs(normalized-row.test_chi2_per_point))
for row in full.itertuples(index=False):
    a = np.loadtxt(ROOT/"data"/f"{row.galaxy}_rotmod.dat")
    expected_bic = row.chi2 + 2*np.log(len(a)) + np.log(2*np.pi*a[:, 2]**2).sum()
    assert np.isclose(expected_bic, row.bic, rtol=1e-12, atol=1e-9)
for row in splits.itertuples(index=False):
    source = ROOT/"data"/f"{row.galaxy}_rotmod.dat"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row.source_sha256

cross_audit = {}
if OUT.name != "wide_bounds" and args.reference_fits is not None:
    old = pd.read_csv(args.reference_fits)
    for model in ("NFW", "Burkert"):
        new_rows = full[full.model == model+" + baryons"].set_index("galaxy")
        old_rows = old[old.model == f"Baryons + {model} (fixed Υ)"].set_index("galaxy")
        delta = new_rows.chi2-old_rows.chi2
        maximum = float(delta.abs().max())
        assert maximum < .001
        cross_audit[model] = maximum
result = dict(full_fits_checked=len(full), heldout_fits_checked=len(held), source_hashes_checked=len(splits),
              maximum_roundtrip_scoring_error=maximum_metric_error, quoted_error_bic_recomputed=True,
              independent_reference_compared=args.reference_fits is not None and OUT.name != "wide_bounds",
              maximum_chi2_difference_from_previous_independent_audit=cross_audit)

# Targeted optimizer regression: a compact Lane-Emden core creates additional
# local minima that an ordinary converged multistart fit can miss.
sys.path.insert(0, str(ROOT/"experiments"))
import dynamics_experiment as dynamics
dynamics.WIDE_BOUNDS = OUT.name == "wide_bounds"
recovery = {}
for density, sigma in ((11.5, 150.), (4.2, 250.), (8.5, 60.)):
    a = np.zeros((60, 8))
    a[:, 0] = np.geomspace(.02, 100., 60)
    a[:, 2] = 1.
    true_p = np.array([density, np.log10(sigma)])
    a[:, 1] = dynamics.prediction(a, true_p, dynamics.MODELS[0])
    fitted = dynamics.fit_curve(a, dynamics.MODELS[0])
    error = float(np.max(np.abs(fitted["p"]-true_p)))
    assert error < 1e-5
    recovery[f"logrho{density}_sigma{sigma}"] = error
result["targeted_fluid_parameter_recovery_errors"] = recovery

dynamics.FLUID_GRID_SIZE = 840
resolution_check = {}
for galaxy, scope in (("NGC2841", "full"), ("NGC4214", "training"), ("UGC02885", "training"), ("UGC09992", "full")):
    table = full if scope == "full" else held
    saved = table[(table.galaxy == galaxy) & (table.model == dynamics.MODELS[0])].iloc[0]
    a = dynamics.load_curve(ROOT/"data"/f"{galaxy}_rotmod.dat")[:int(saved.n)]
    refined = dynamics.fit_curve(a, dynamics.MODELS[0])
    error = abs(float(saved.chi2)-refined["chi2"])
    assert error < 1e-5
    resolution_check[f"{galaxy}_{scope}"] = error
result["doubling_profile_grid_max_chi2_change_by_case"] = resolution_check
(OUT/"validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
