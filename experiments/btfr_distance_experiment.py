#!/usr/bin/env python3
"""Independent-distance leave-one-galaxy-out BTFR feasibility benchmark.

Uses existing SPARC table, not raw new IR/HI photometry. It is conditional on
mass-to-light, HI calibration and independently measured inclinations/speeds.
No calibration from Hubble-flow distances and no image-only claims.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/solid_galaxy_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.kinematic_distance import baryonic_flux_proxy, BaryonicTullyFisher
OUT = ROOT / "executed_solid" / "btfr"
SOURCE = ROOT / "executed_research/images/SPARC_Lelli2016c.mrt"


def read_table():
    names = ["galaxy", "type", "distance_mpc", "distance_error_mpc", "distance_method",
        "inclination_deg", "inclination_error_deg", "luminosity_1e9_lsun", "luminosity_error_1e9_lsun",
        "reff_kpc", "surface_brightness_eff", "disk_scale_kpc", "surface_brightness_disk",
        "hi_mass_1e9_msun", "hi_radius_kpc", "vflat_kms", "vflat_error_kms", "quality"]
    rows = []
    # Numeric fields are whitespace-separated in the supplied MRT; its published
    # byte offsets do not parse this particular text copy correctly.
    for line in SOURCE.read_text().splitlines():
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            values = [parts[0]] + [float(x) for x in parts[1:18]]
        except ValueError:
            continue
        rows.append(dict(zip(names, values)))
    frame = pd.DataFrame(rows)
    if len(frame) != 175 or frame.galaxy.duplicated().any():
        raise ValueError("Expected all 175 distinct SPARC table rows")
    return frame


def metrics(frame, pred):
    frac = pred / frame.distance_mpc.to_numpy() - 1
    residual = np.log10(pred / frame.distance_mpc.to_numpy())
    return {"galaxies": len(frame), "median_absolute_fractional_error": float(np.median(abs(frac))),
        "mean_absolute_fractional_error": float(np.mean(abs(frac))),
        "mean_signed_fractional_error": float(np.mean(frac)),
        "rms_log10_distance_error_dex": float(np.sqrt(np.mean(residual**2)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = read_table()
    keep = (table.distance_method.isin([2, 3]) & (table.vflat_kms > 0)
        & (table.inclination_deg >= 30) & (table.quality <= 2)
        & (table.luminosity_1e9_lsun > 0) & (table.hi_mass_1e9_msun >= 0))
    sample = table.loc[keep].copy().reset_index(drop=True)
    protocol = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "distance_labels": "SPARC f_D=2 TRGB or f_D=3 Cepheid only",
        "quality": "Q<=2, inclination>=30deg, positive Vflat and luminosity",
        "split": "leave-one-galaxy-out; no held-out label enters slope/intercept calibration",
        "mass_to_light_values": [.5, .35, .7], "primary_mass_to_light": .5,
        "helium_factor": 1.33, "slope": "OLS fitted; fixed-four sensitivity also predeclared",
        "observables": "C=(0.5*L36+1.33*MHI)/D_reference² and independently measured Vflat",
        "flux_proxy_warning": "Derived from rounded catalog luminosities/masses, not a new raw-flux reduction. The reference D cancels algebraically if those quantities consistently scale as D².",
        "uncertainty": "No new independently calibrated individual distance interval; common systematics and calibration errors not modeled."}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    predictions, summary = [], {}
    for upsilon in protocol["mass_to_light_values"]:
        coefficient = baryonic_flux_proxy(sample.luminosity_1e9_lsun.to_numpy() * 1e9,
            sample.hi_mass_1e9_msun.to_numpy() * 1e9, sample.distance_mpc.to_numpy(), upsilon)
        mass = coefficient * sample.distance_mpc.to_numpy() ** 2
        for slope in (None, 4.):
            label = f"upsilon_{upsilon:g}_" + ("fitted_slope" if slope is None else "slope4")
            predicted, candle, fits = [], [], []
            for index, row in sample.iterrows():
                train = np.arange(len(sample)) != index
                model = BaryonicTullyFisher.fit(sample.vflat_kms.to_numpy()[train], mass[train], slope)
                estimate = float(model.distance(row.vflat_kms, coefficient[index]))
                baseline = float(np.sqrt(10**np.median(np.log10(mass[train])) / coefficient[index]))
                predicted.append(estimate); candle.append(baseline); fits.append(model)
                predictions.append({"galaxy": row.galaxy, "configuration": label,
                    "distance_method": int(row.distance_method), "reference_label_mpc": row.distance_mpc,
                    "reference_label_error_mpc": row.distance_error_mpc,
                    "baryon_flux_msun_per_mpc2": coefficient[index], "vflat_kms": row.vflat_kms,
                    "vflat_error_kms": row.vflat_error_kms, "predicted_mpc": estimate,
                    "flux_only_standard_candle_mpc": baseline, "trained_without_this_galaxy": True,
                    "training_galaxies": model.calibration_galaxies, "slope": model.slope,
                    "intercept_at_100kms": model.intercept_log_msun_at_100kms})
            summary[label] = {"btfr": metrics(sample, np.array(predicted)),
                "flux_only_standard_candle": metrics(sample, np.array(candle)),
                "fit_slope_range": [min(m.slope for m in fits), max(m.slope for m in fits)]}
    sample.to_csv(OUT / "independent_distance_sample.csv", index=False)
    table.loc[~keep].to_csv(OUT / "excluded_sample.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "leave_one_galaxy_out_predictions.csv", index=False)
    result = {"protocol": protocol, "eligible_galaxies": len(sample),
        "distance_method_counts": {str(k): int(v) for k, v in sample.distance_method.value_counts().items()},
        "configurations": summary, "primary": summary["upsilon_0.5_fitted_slope"],
        "interpretation": "An externally calibrated relation plus independent kinematics can supply a distance scale. This experiment neither requires nor identifies viscosity and is not an image-only distance estimate.",
        "limitations": ["Only 30 suitable local calibrators; shared method/group/calibration systematics remain.",
            "LOO test predictions share overlapping training sets; errors are not independent retraining experiments.",
            "Fixed total stellar M/L, helium factor, inclination correction, and no molecular-gas correction.",
            "OLS does not model heteroscedastic velocity and distance errors; no precision distance-ladder claim."]}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    primary = pd.DataFrame(predictions).query("configuration == 'upsilon_0.5_fitted_slope'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.scatter(primary.reference_label_mpc, primary.flux_only_standard_candle_mpc, label="Flux-only standard candle", alpha=.6)
    ax.scatter(primary.reference_label_mpc, primary.predicted_mpc, label="BTFR + independent Vflat", alpha=.8)
    ax.plot([1, 60], [1, 60], "k--", lw=1); ax.set(xscale="log", yscale="log",
        xlabel="Independent TRGB/Cepheid distance (Mpc)", ylabel="Leave-one-galaxy-out prediction (Mpc)")
    ax.legend(fontsize=8)
    axes[1].scatter(primary.vflat_kms, primary.predicted_mpc / primary.reference_label_mpc - 1)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="Independently measured Vflat (km/s)", ylabel="Fractional distance residual")
    fig.suptitle("Calibrated dynamical distance feasibility; not imaging-only or fluid evidence")
    fig.savefig(OUT / "distance_validation.png", dpi=160); plt.close(fig)
    print(json.dumps(result["primary"], indent=2))


if __name__ == "__main__":
    main()
