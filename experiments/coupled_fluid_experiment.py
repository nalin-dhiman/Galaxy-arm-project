"""Execute coupled spherical validation and a separate real-image/kinematics link.

Run from repository root:
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/galaxy-mpl python experiments/coupled_fluid_experiment.py

The exact spherical experiment uses known synthetic baryons. Real SPARC disk
fits retain the explicitly labeled uncontracted approximation. Images define
angular measurement windows only; external catalog distances convert them to
kpc. Images never provide heldout velocities, a viscosity, or a distance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.coupled_fluid import CoupledSphericalFluidHalo, HernquistBaryons, PlummerBaryons
from galaxy_lab.fluid import G, IsothermalFluidHalo
from experiments.dynamics_experiment import (
    MODELS, baryonic_v2, fit_curve, fit_record, load_curve, outer_split, prediction,
)

TARGETS = ("NGC2403", "NGC3198", "NGC5055")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def max_relative(a, b):
    return float(np.max(np.abs(a-b)/np.maximum(np.abs(b), 1e-30)))


def validate_spherical(output):
    checks = []
    rr = np.geomspace(.003, 120., 170)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []
    for kind in (PlummerBaryons, HernquistBaryons):
        b = kind(4e10, 2.)
        h = CoupledSphericalFluidHalo(3e7, 90., b, 150.)
        eps = rr*1e-4
        dp = (h.pressure(rr+eps)-h.pressure(rr-eps))/(2*eps)
        dm = (h.enclosed_mass(rr+eps)-h.enclosed_mass(rr-eps))/(2*eps)
        dphi = (h.potential_offset(rr+eps)-h.potential_offset(rr-eps))/(2*eps)
        force = G*h.total_mass(rr)/rr**2
        alone = IsothermalFluidHalo(3e7, 90.)
        checks.append(dict(baryon_profile=kind.__name__, baryon_mass_msun=b.mass_msun,
                           baryon_scale_kpc=b.scale_kpc,
                           hydrostatic_max_relative_residual=max_relative(dp, -h.density(rr)*force),
                           dm_mass_continuity_max_relative_residual=max_relative(dm, 4*np.pi*rr**2*h.density(rr)),
                           total_potential_gradient_max_relative_residual=max_relative(dphi, force),
                           coupled_to_isolated_density_ratio_at_2kpc=float(h.density(2.)/alone.density(2.))))
        axes[0].loglog(rr, h.density(rr), label=f"Coupled {kind.__name__.replace('Baryons','')}")
        axes[1].semilogx(rr, h.circular_velocity(rr), label=f"Coupled {kind.__name__.replace('Baryons','')}")
        uncontracted = np.sqrt(alone.circular_velocity(rr)**2+G*b.enclosed_mass(rr)/rr)
        axes[1].semilogx(rr, uncontracted, "--", alpha=.7,
                        label=f"Isolated halo + {kind.__name__.replace('Baryons','')} force")
        for j, r in enumerate(rr):
            rows.append(dict(baryon_profile=kind.__name__, r_kpc=float(r),
                             coupled_dark_density_msun_kpc3=float(h.density(r)),
                             isolated_dark_density_msun_kpc3=float(alone.density(r)),
                             coupled_total_velocity_kms=float(h.circular_velocity(r)),
                             uncontracted_total_velocity_kms=float(uncontracted[j])))
    axes[0].loglog(rr, alone.density(rr), "k:", label="Isolated dark halo")
    axes[0].set(xlabel="Radius [kpc]", ylabel="Dark density [Msun/kpc³]")
    axes[1].set(xlabel="Radius [kpc]", ylabel="Total circular speed [km/s]")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle("Exact spherical baryon coupling at fixed central dark density and dispersion")
    fig.tight_layout()
    fig.savefig(output/"spherical_baryon_response.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output/"spherical_baryon_response.csv", index=False)
    zero = CoupledSphericalFluidHalo(3e7, 90., PlummerBaryons(0., 2.), 150.)
    zero_error = max_relative(zero.circular_velocity(rr), alone.circular_velocity(rr))
    scale = 3.
    h = CoupledSphericalFluidHalo(3e7, 90., HernquistBaryons(4e10, 2.), 150.)
    hs = CoupledSphericalFluidHalo(3e7/scale**2, 90., HernquistBaryons(4e10*scale, 2.*scale), 150.*scale)
    degeneracy = max_relative(h.circular_velocity(rr), hs.circular_velocity(rr*scale))
    assert zero_error < 1e-7
    assert degeneracy < 1e-8
    for row in checks:
        assert row["hydrostatic_max_relative_residual"] < 1e-5
        assert row["dm_mass_continuity_max_relative_residual"] < 1e-5
        assert row["total_potential_gradient_max_relative_residual"] < 1e-5
    result = dict(physical_checks=checks, zero_baryon_max_relative_speed_error=zero_error,
                  spherical_distance_rescaling_max_relative_speed_error=degeneracy,
                  distance_rescaling_rule="r,a -> lambda(r,a); rho0 -> rho0/lambda²; M_b -> lambda M_b; sigma fixed",
                  interpretation="Scale invariance persists if baryonic masses are not independently calibrated.")
    dump(output/"coupled_closure_checks.json", result)
    return result


def synthetic_known_baryons(output):
    # Predeclared, reproducible numerical demonstration, not an observed galaxy.
    rng = np.random.default_rng(20261023)
    r = np.geomspace(.3, 35., 50)
    baryons = HernquistBaryons(2e10, 1.8)
    rho0, sigma = 6e7, 110.
    truth = CoupledSphericalFluidHalo(rho0, sigma, baryons, 50.)
    actual = truth.circular_velocity(r)
    error = np.full(len(r), 3.)
    observed = actual+rng.normal(0., error)
    split = int(.7*len(r))
    def pred(p, radii, coupled):
        if coupled:
            return CoupledSphericalFluidHalo(10.**p[0], 10.**p[1], baryons, 50.).circular_velocity(radii)
        halo = IsothermalFluidHalo(10.**p[0], 10.**p[1])
        return np.sqrt(halo.circular_velocity(radii)**2+G*baryons.enclosed_mass(radii)/radii)
    results, all_rows = [], []
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(r, observed, error, fmt=".", color="gray", label="Synthetic observations")
    ax.plot(r, actual, "k--", label="Known coupled spherical truth")
    ax.axvline(np.sqrt(r[split-1]*r[split]), color="gray", linestyle=":", label="Training / heldout boundary")
    for coupled in (True, False):
        candidates = []
        for start in ([7., 2.], [8., 2.2], [6.5, 2.3]):
            opt = least_squares(lambda p: (pred(p, r[:split], coupled)-observed[:split])/error[:split],
                                start, bounds=([4., 1.], [10., 2.7]), max_nfev=150,
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            if opt.success:
                candidates.append(opt)
        if not candidates:
            raise RuntimeError("Synthetic optimization failed")
        best = min(candidates, key=lambda z: np.sum(z.fun**2))
        predicted = pred(best.x, r, coupled)
        name = "Coupled spherical baryons + halo" if coupled else "Uncontracted spherical force sum"
        rec = dict(model=name, rho0_msun_kpc3=float(10.**best.x[0]), sigma_kms=float(10.**best.x[1]),
                   training_rmse_kms=float(np.sqrt(np.mean((predicted[:split]-observed[:split])**2))),
                   heldout_rmse_kms=float(np.sqrt(np.mean((predicted[split:]-observed[split:])**2))),
                   truth_heldout_rmse_kms=float(np.sqrt(np.mean((predicted[split:]-actual[split:])**2))),
                   success=True, n_parameters=2)
        results.append(rec)
        for j, radius in enumerate(r):
            all_rows.append(dict(model=name, radius_kpc=float(radius), true_speed_kms=float(actual[j]),
                                 observed_speed_kms=float(observed[j]), error_kms=float(error[j]),
                                 predicted_speed_kms=float(predicted[j]), heldout=bool(j >= split)))
        ax.plot(r, predicted, label=name)
    ax.set(xscale="log", xlabel="Radius [kpc]", ylabel="Circular speed [km/s]",
           title="Known-baryon synthetic validation; two fitted dark parameters")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"synthetic_coupled_holdout.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(all_rows).to_csv(output/"synthetic_predictions.csv", index=False)
    result = dict(seed=20261023, true_rho0_msun_kpc3=rho0, true_sigma_kms=sigma,
                  fixed_known_baryon_mass_msun=baryons.mass_msun, fixed_known_baryon_scale_kpc=baryons.scale_kpc,
                  n_train=split, n_test=len(r)-split, results=results,
                  interpretation="Implementation demonstration with simulated data; not evidence for a real galaxy or fluid microphysics.")
    dump(output/"synthetic_fit_summary.json", result)
    return result


def matched_sparc(output, windows_path):
    meta = pd.read_csv(ROOT/"executed_research/images/sparc_metadata.csv").set_index("galaxy")
    windows = pd.read_csv(windows_path) if windows_path.exists() else pd.DataFrame()
    records, forecasts, window_rows, file_rows = [], [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, galaxy in zip(axes, TARGETS):
        path = ROOT/f"data/{galaxy}_rotmod.dat"
        a = load_curve(path)
        split, status = outer_split(len(a))
        assert status == "eligible"
        m = meta.loc[galaxy]
        distance = float(m.distance_mpc)
        kpc_per_arcsec = distance*1000.*np.pi/(180*3600)
        angular = a[:, 0]/kpc_per_arcsec
        for f in (path, ROOT/f"executed_research/images/{galaxy}_r.fits"):
            file_rows.append(dict(path=str(f.relative_to(ROOT)), sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        ax.errorbar(a[:, 0], a[:, 1], a[:, 2], fmt=".", color="k", label="Independent SPARC velocity")
        ax.axvline(.5*(a[split-1, 0]+a[split, 0]), color="gray", linestyle=":")
        for model in MODELS:
            full = fit_curve(a, model)
            trained = fit_curve(a[:split], model)
            assert full["success"] and trained["success"]
            for scope, fitted, fit_array in (("all radii", full, a), ("inner-only training", trained, a[:split])):
                rec = fit_record(galaxy, fit_array, model, fitted, scope)
                rec.pop("optimality", None)
                rec["distance_mpc_external"] = distance
                rec["inclination_deg_external"] = float(m.inclination_deg)
                rec["n_heldout"] = len(a)-split
                if scope == "inner-only training":
                    testpred = prediction(a[split:], fitted["p"], model)
                    rec["heldout_rmse_kms"] = float(np.sqrt(np.mean((testpred-a[split:, 1])**2)))
                records.append(rec)
            predicted = prediction(a, trained["p"], model)
            ax.plot(a[:, 0], predicted, label=model)
            for j, r in enumerate(a[:, 0]):
                row = dict(galaxy=galaxy, model=model, radius_kpc=float(r), radius_arcsec=float(angular[j]),
                           distance_mpc_external=distance, observed_velocity_kms=float(a[j, 1]),
                           error_kms=float(a[j, 2]), predicted_velocity_kms=float(predicted[j]),
                           heldout=bool(j >= split), baryon_v2_signed=float(baryonic_v2(a)[j]))
                if model == MODELS[0]:
                    halo = IsothermalFluidHalo(*10.**trained["p"])
                    row.update(dark_density_msun_kpc3=float(halo.density(r)),
                               dark_mass_msun=float(halo.enclosed_mass(r)),
                               dispersion_kms=halo.sigma_kms)
                forecasts.append(row)
            if not windows.empty:
                for _, window in windows[windows.galaxy == galaxy].iterrows():
                    inner, outer = float(window.inner_radius_arcsec), float(window.outer_radius_arcsec)
                    inside = (angular >= inner) & (angular <= outer)
                    for kind, selector in (("all measured radii", inside), ("heldout radii only", inside & (np.arange(len(a)) >= split))):
                        row = dict(galaxy=galaxy, model=model, window_scope=kind,
                                   inner_radius_arcsec=inner, outer_radius_arcsec=outer,
                                   inner_radius_kpc=inner*kpc_per_arcsec, outer_radius_kpc=outer*kpc_per_arcsec,
                                   n_radii=int(selector.sum()), accepted=window.get("accepted", False),
                                   reason=window.get("reason", ""), distance_mpc_external=distance,
                                   radius_uncertainty_fraction_from_catalog_distance=float(m.distance_error_mpc/m.distance_mpc))
                        if selector.any():
                            row["rmse_kms"] = float(np.sqrt(np.mean((predicted[selector]-a[selector, 1])**2)))
                        window_rows.append(row)
        if not windows.empty:
            for _, window in windows[windows.galaxy == galaxy].iterrows():
                ax.axvspan(float(window.inner_radius_arcsec)*kpc_per_arcsec,
                           float(window.outer_radius_arcsec)*kpc_per_arcsec, alpha=.1, color="teal")
        ax.set(xlabel="Radius [kpc, external catalog distance]", ylabel="Speed [km/s]", title=f"{galaxy}: D = {distance:g} Mpc")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.suptitle("Independent image + rotation-curve matches: inner-trained predictions; image annuli shaded")
    fig.tight_layout()
    fig.savefig(output/"matched_sparc_predictions.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output/"matched_sparc_fits.csv", index=False)
    pd.DataFrame(forecasts).to_csv(output/"matched_sparc_predictions.csv", index=False)
    if window_rows:
        pd.DataFrame(window_rows).to_csv(output/"image_window_kinematics.csv", index=False)
    dump(output/"matched_input_hashes.json", file_rows)
    fluid = [r for r in records if r["model"] == MODELS[0] and r["scope"] == "inner-only training"]
    result = dict(targets=list(TARGETS), exact_spherical_model_applied_to_real_disks=False,
                  real_disk_model="Uncontracted isolated isothermal halo + signed equatorial SPARC baryon forces",
                  stellar_mass_to_light_disk=.5, stellar_mass_to_light_bulge=.7,
                  windows_available=not windows.empty, windows_path=str(windows_path.relative_to(ROOT)),
                  image_windows_evaluated=len(window_rows),
                  fluid_heldout=[dict(galaxy=r["galaxy"], rmse_kms=r["heldout_rmse_kms"],
                                      distance_mpc_external=r["distance_mpc_external"],
                                      bound_hit=bool(r["bound_hit"])) for r in fluid],
                  limitations=["Only three convenience matches, not a population claim.",
                               "Catalog D and inclination are supplied externally; errors are not marginalized in the fits.",
                               "Optical image shape/annulus does not infer dark density without independent kinematics and physical scale.",
                               "Measurement annuli are not independently validated traced arms unless explicitly recorded as such.",
                               "Static equilibrium supplies no fitted viscosity.",
                               "Image windows inside training range are in-sample links, not heldout velocity predictions."])
    dump(output/"matched_sparc_summary.json", result)
    return result


def image_window_holdout(output, windows_path):
    """Fit outside image-defined annuli, predict independently measured inside.

    Windows were declared from images without reference to velocity values.
    This is conditional kinematic interpolation, separate from the original
    inner-training/outer-test extrapolation. Baryon templates and external D
    remain known covariates. No arm-tracing validation is implied.
    """
    if not windows_path.exists():
        return dict(status="not run", reason="No independently declared image windows available")
    windows = pd.read_csv(windows_path)
    meta = pd.read_csv(ROOT/"executed_research/images/sparc_metadata.csv").set_index("galaxy")
    records, forecasts, skipped = [], [], []
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 4.5), squeeze=False)
    for ax, galaxy in zip(axes[0], TARGETS):
        a = load_curve(ROOT/f"data/{galaxy}_rotmod.dat")
        distance = float(meta.loc[galaxy, "distance_mpc"])
        kpc_per_arcsec = distance*1000.*np.pi/(180*3600)
        angular = a[:, 0]/kpc_per_arcsec
        chosen = windows[windows.galaxy == galaxy]
        ax.set(title=galaxy, xlabel="Radius [kpc, external catalog distance]", ylabel="Speed [km/s]")
        for window_index, (_, window) in enumerate(chosen.iterrows()):
            inner, outer = float(window.inner_radius_arcsec), float(window.outer_radius_arcsec)
            test = (angular >= inner) & (angular <= outer)
            train = ~test
            base = dict(galaxy=galaxy, window_index=window_index,
                        inner_radius_arcsec=inner, outer_radius_arcsec=outer,
                        inner_radius_kpc=inner*kpc_per_arcsec, outer_radius_kpc=outer*kpc_per_arcsec,
                        n_train=int(train.sum()), n_test=int(test.sum()),
                        distance_mpc_external=distance, accepted_arm_trace=bool(window.get("accepted", False)),
                        window_reason=window.get("reason", ""),
                        split_type="Fit outside image annulus; predict inside (conditional interpolation)")
            if train.sum() < 5 or test.sum() < 2:
                skipped.append(dict(**base, reason="Requires at least 5 training and 2 test velocities"))
                continue
            ax.errorbar(a[train, 0], a[train, 1], a[train, 2], fmt=".", color="gray", label="Training: outside image annulus")
            ax.errorbar(a[test, 0], a[test, 1], a[test, 2], fmt="o", markersize=3, color="k", label="Test: inside image annulus")
            ax.axvspan(inner*kpc_per_arcsec, outer*kpc_per_arcsec, color="teal", alpha=.1)
            for model in MODELS:
                try:
                    result = fit_curve(a[train], model)
                except (ValueError, RuntimeError) as exc:
                    records.append(dict(**base, model=model, success=False, reason=str(exc)))
                    continue
                if not result["success"]:
                    records.append(dict(**base, model=model, success=False, reason="Optimizer did not produce a valid physical fit"))
                    continue
                predicted = prediction(a, result["p"], model)
                if not np.isfinite(predicted).all():
                    records.append(dict(**base, model=model, success=False,
                                        reason="Model invalid at one or more prediction radii"))
                    continue
                rec = fit_record(galaxy, a[train], model, result, "image-annulus exterior training")
                rec.pop("optimality", None)
                rec.update(base)
                rec.update(heldout_rmse_kms=float(np.sqrt(np.mean((predicted[test]-a[test, 1])**2))),
                           heldout_mae_kms=float(np.mean(np.abs(predicted[test]-a[test, 1]))),
                           heldout_chi2=float(np.sum(((predicted[test]-a[test, 1])/a[test, 2])**2)),
                           success=True)
                records.append(rec)
                ax.plot(a[:, 0], predicted, label=model)
                for j, radius in enumerate(a[:, 0]):
                    forecasts.append(dict(galaxy=galaxy, model=model, window_index=window_index,
                                          radius_kpc=float(radius), radius_arcsec=float(angular[j]),
                                          observed_velocity_kms=float(a[j, 1]), error_kms=float(a[j, 2]),
                                          predicted_velocity_kms=float(predicted[j]), heldout=bool(test[j]),
                                          split_type=base["split_type"]))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Image-defined regional holdout: fit outside shaded annuli, predict measured velocities inside")
    fig.tight_layout()
    fig.savefig(output/"image_window_holdout.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(records).to_csv(output/"image_window_holdout_fits.csv", index=False)
    pd.DataFrame(forecasts).to_csv(output/"image_window_holdout_predictions.csv", index=False)
    if skipped:
        dump(output/"image_window_holdout_skipped.json", skipped)
    result = dict(status="executed", windows_declared_from_images=True,
                  n_model_fits=len(records), n_success=sum(bool(r["success"]) for r in records),
                  skipped_windows=skipped,
                  models=[dict(galaxy=r["galaxy"], model=r["model"], success=bool(r["success"]),
                               n_train=r["n_train"], n_test=r["n_test"],
                               heldout_rmse_kms=r.get("heldout_rmse_kms"),
                               bound_hit=bool(r.get("bound_hit", False))) for r in records],
                  limitations=["Fits use independent measured velocities outside each image window, plus known baryon templates and catalog D.",
                               "This tests conditional prediction of velocities in an image-selected region; it is not image-only velocity inference.",
                               "Both smaller and larger radii can occur in training: this differs from the outer extrapolation experiment.",
                               "No named target's image window has an independently validated arm trace.",
                               "Three convenience galaxies cannot establish a population model ranking."])
    dump(output/"image_window_holdout_summary.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT/"executed_filters/fluid")
    ap.add_argument("--windows", type=Path, default=ROOT/"executed_filters/morphology/arm_windows.csv")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(spherical_validation=validate_spherical(args.output),
                  synthetic_known_baryons=synthetic_known_baryons(args.output),
                  matched_real_kinematics=matched_sparc(args.output, args.windows),
                  image_window_holdout=image_window_holdout(args.output, args.windows))
    dump(args.output/"summary.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
