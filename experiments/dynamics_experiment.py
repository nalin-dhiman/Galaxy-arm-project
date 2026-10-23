"""Predeclared two-parameter rotation-curve comparison and outer-radius test.

Run from the project root:
  OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/galaxy-mpl python experiments/dynamics_experiment.py

Input velocities are never differentiated. For the extrapolation test only the
inner training velocities initialize and fit parameters. Outer baryonic force
templates remain known covariates, and test velocities are used only for scoring.
The fluid halo is hydrostatic/isothermal and its baryons are added without halo
contraction: this does not fit viscosity, an interaction rate, or a distance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_lab.fluid import dimensionless_profile, fluid_circular_speed  # noqa: E402

G = 4.30091727003628e-6  # kpc (km/s)^2 / Msun
OUT = ROOT / "executed_research" / "dynamics"
MODELS = ("Isothermal fluid + baryons", "NFW + baryons", "Burkert + baryons", "Exponential curve")
K = 2
WIDE_BOUNDS = False
FLUID_GRID_SIZE = 420


def load_curve(path: Path) -> np.ndarray:
    a = np.atleast_2d(np.loadtxt(path))
    good = np.isfinite(a[:, :6]).all(axis=1) & (a[:, 0] > 0) & (a[:, 2] > 0)
    return a[good][np.argsort(a[good, 0], kind="stable")]


def baryonic_v2(a: np.ndarray) -> np.ndarray:
    return a[:, 3] * np.abs(a[:, 3]) + .5 * a[:, 4] ** 2 + .7 * a[:, 5] ** 2


def halo_v2(r: np.ndarray, p: np.ndarray, model: str) -> np.ndarray:
    if model == MODELS[0]:
        return np.asarray(fluid_circular_speed(r, p[0], p[1])) ** 2
    rho, rs = 10. ** p[0], 10. ** p[1]
    x = r / rs
    if model == MODELS[1]:
        term = np.log1p(x) - x / (1. + x)
        small = x < 1e-3
        xs = x[small]
        term[small] = xs**2 / 2 - 2*xs**3/3 + 3*xs**4/4 - 4*xs**5/5 + 5*xs**6/6
        return G * 4 * np.pi * rho * rs**3 * term / r
    term = np.log1p(x) + .5 * np.log1p(x*x) - np.arctan(x)
    small = x < 1e-3
    xs = x[small]
    term[small] = 2*xs**3/3 - xs**4/2 + 2*xs**7/7 - xs**8/4
    return G * 2 * np.pi * rho * rs**3 * term / r


def raw_v2(a: np.ndarray, p: np.ndarray, model: str) -> np.ndarray:
    return baryonic_v2(a) + halo_v2(a[:, 0], p, model)


def prediction(a: np.ndarray, p: np.ndarray, model: str) -> np.ndarray:
    if model == MODELS[-1]:
        amplitude, scale = 10. ** p[0], 10. ** p[1]
        return amplitude * -np.expm1(-a[:, 0] / scale)
    v2 = raw_v2(a, p, model)
    # Negative total radial force is invalid, never silently zero-clipped.
    if np.any(v2 < 0) or not np.isfinite(v2).all():
        return np.full(len(a), np.nan)
    return np.sqrt(v2)


def bounds_for(model: str) -> tuple[np.ndarray, np.ndarray]:
    if model == MODELS[0]:
        return (np.array([-8., 0.]), np.array([14., math.log10(3000.)])) if WIDE_BOUNDS else (np.array([-6., 0.]), np.array([12., math.log10(600.)]))
    if model == MODELS[-1]:
        return (np.array([0., -4.]), np.array([3., 5.])) if WIDE_BOUNDS else (np.array([0., -3.]), np.array([math.log10(600.), 3.]))
    return (np.array([-8., -4.]), np.array([14., 5.])) if WIDE_BOUNDS else (np.array([-6., -3.]), np.array([12., 3.]))


def starts_for(a: np.ndarray, model: str) -> list[list[float]]:
    # Every data-derived quantity here uses only the array passed for fitting.
    r, v = a[:, 0], a[:, 1]
    if model == MODELS[0]:
        residual_v2 = np.maximum(v**2 - baryonic_v2(a), 1.)
        sigma = np.clip(np.sqrt(np.median(residual_v2) / 2), 2., 500.)
        starts = [[rho, math.log10(sigma)] for rho in (5., 7., 9.)]
        starts += [[6., math.log10(max(2., sigma / 2))], [8., math.log10(min(550., sigma * 2))]]
        return starts
    return [[6., 0.], [7., 1.], [8., 0.], [5., 2.], [9., -1.]]


def fit_profiled_fluid(a: np.ndarray) -> dict:
    """Profile convex sigma² amplitude and scan/refine the remaining scale.

    At fixed r0, v²=B+q*w(r/r0), q=sigma². For nonnegative observed velocity,
    d²chi²/dq²=sum(v*w²/[2*err²*(B+q*w)^(3/2)]) >= 0. Thus a bracketed
    derivative root gives the global amplitude optimum, including bound cases.
    This is a derivative of the likelihood, never a derivative of target data.
    """
    if np.any(a[:, 1] < 0):
        raise ValueError("Convex amplitude profiling requires nonnegative target speeds")
    lo, hi = bounds_for(MODELS[0])
    r, v, err = a[:, 0], a[:, 1], a[:, 2]
    baryons = baryonic_v2(a)
    c = math.log10(4*np.pi*G)
    log_r0_bounds = (lo[1]-.5*c-.5*hi[0], hi[1]-.5*c-.5*lo[0])
    evaluations = 0

    def profile(log_r0: float) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        _, shape = dimensionless_profile(r/10.**log_r0)
        log_q_lo = max(2*lo[1], lo[0]+c+2*log_r0)
        log_q_hi = min(2*hi[1], hi[0]+c+2*log_r0)
        # Enforce nonnegative total force, including signed gas contributions.
        needed = max(0., float(np.max(-baryons/shape)))
        if needed > 0:
            log_q_lo = max(log_q_lo, math.log10(needed*(1+1e-12)+1e-12))
        if log_q_lo > log_q_hi+1e-11:
            return float("inf"), np.full(2, np.nan)
        log_q_lo = min(log_q_lo, log_q_hi)

        def derivative(log_q: float) -> float:
            v2 = baryons+10.**log_q*shape
            if np.min(v2) <= 0:
                return -float("inf")
            return float(np.sum(shape*(1-v/np.sqrt(v2))/err**2))

        if derivative(log_q_lo) >= 0:
            log_q = log_q_lo
        elif derivative(log_q_hi) <= 0:
            log_q = log_q_hi
        else:
            log_q = brentq(derivative, log_q_lo, log_q_hi, xtol=1e-12)
        p = np.clip(np.array([log_q-c-2*log_r0, .5*log_q]), lo, hi)
        pred = prediction(a, p, MODELS[0])
        if not np.isfinite(pred).all():
            return float("inf"), np.full(2, np.nan)
        return float(np.sum(((pred-v)/err)**2)), p

    # Lane-Emden oscillations have broad logarithmic periods; this grid spans
    # every allowed core radius, then independently refines each local minimum.
    grid = np.linspace(*log_r0_bounds, FLUID_GRID_SIZE)
    profiled = [profile(value) for value in grid]
    costs = np.array([entry[0] for entry in profiled])
    candidates = [entry for entry in (profiled[0], profiled[-1]) if np.isfinite(entry[0])]
    for j in range(1, len(grid)-1):
        if np.isfinite(costs[j]) and costs[j] <= costs[j-1] and costs[j] <= costs[j+1]:
            result = minimize_scalar(lambda z: profile(z)[0], bounds=(grid[j-1], grid[j+1]),
                                     method="bounded", options={"xatol": 1e-11})
            if result.success:
                candidates.append(profile(result.x))
    # At a force-feasibility edge only one grid point may be finite. Retain
    # every finite grid minimum as well, ensuring a result can never be lost.
    candidates.append(profiled[int(np.argmin(costs))])
    candidates = [entry for entry in candidates if np.isfinite(entry[0])]
    if not candidates:
        raise RuntimeError("No physically feasible fluid amplitude/scale pair")
    cost, p = min(candidates, key=lambda entry: entry[0])
    return dict(p=p, success=True, chi2=cost, optimizer="convex amplitude profile + global radius grid",
                n_starts=len(grid), n_converged=len(candidates), n_valid=len(candidates),
                min_total_v2=float(np.min(raw_v2(a, p, MODELS[0]))),
                bound_hit=bool(np.any(np.minimum(p-lo, hi-p) < 1e-4)), nfev=evaluations, optimality=float("nan"))


def fit_curve(a: np.ndarray, model: str) -> dict:
    if model == MODELS[0]:
        return fit_profiled_fluid(a)
    lo, hi = bounds_for(model)
    v, err = a[:, 1], a[:, 2]
    if model == MODELS[-1]:
        # Profile out amplitude, with all local minima on a deterministic
        # logarithmic scale grid refined independently. No saved full fit seed.
        def profile(log_scale: float) -> tuple[float, np.ndarray]:
            shape = -np.expm1(-a[:, 0] / 10.**log_scale)
            amplitude = np.clip(np.sum(v*shape/err**2) / np.sum(shape**2/err**2), 10.**lo[0], 10.**hi[0])
            p = np.array([math.log10(amplitude), log_scale])
            cost = float(np.sum(((amplitude*shape-v)/err)**2))
            return cost, p
        grid = np.linspace(lo[1], hi[1], 180)
        costs = np.array([profile(z)[0] for z in grid])
        candidates = [profile(grid[0]), profile(grid[-1])]
        for j in range(1, len(grid)-1):
            if costs[j] <= costs[j-1] and costs[j] <= costs[j+1]:
                result = minimize_scalar(lambda x: profile(x)[0], bounds=(grid[j-1], grid[j+1]),
                                         method="bounded", options={"xatol": 1e-11})
                if result.success:
                    candidates.append(profile(result.x))
        cost, p = min(candidates, key=lambda z: z[0])
        return dict(p=p, success=True, chi2=cost, optimizer="profiled bounded search",
                    n_starts=len(candidates), n_converged=len(candidates), n_valid=len(candidates),
                    min_total_v2=float("nan"), bound_hit=bool(np.any(np.minimum(p-lo, hi-p) < 1e-4)),
                    nfev=-1, optimality=float("nan"))

    # Signed square root supplies a continuous optimization extension into an
    # invalid region. Such candidates cannot enter any reported physical fit.
    def residual(p: np.ndarray) -> np.ndarray:
        v2 = raw_v2(a, p, model)
        return (np.sign(v2)*np.sqrt(np.abs(v2)) - v) / err

    candidates, attempts = [], []
    for p0 in starts_for(a, model):
        p0 = np.clip(p0, lo+1e-8, hi-1e-8)
        result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=1600,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, x_scale="jac")
        attempts.append(result)
        v2 = raw_v2(a, result.x, model)
        if np.isfinite(v2).all() and np.min(v2) >= 0 and result.success:
            candidates.append((float(np.sum(result.fun**2)), result.x, result, "least_squares"))
    if not candidates:
        # A physical boundary can matter for signed gas or baryon overshoot.
        # Solve with explicit nonnegative total-force constraints; still no
        # clipping of the final force or acceptance of failed optimizations.
        for attempt in sorted(attempts, key=lambda z: np.sum(z.fun**2))[:3]:
            p0 = attempt.x.copy()
            while np.min(raw_v2(a, p0, model)) < 0 and p0[0] < hi[0] - .1:
                p0[0] += .1
            result = minimize(lambda p: float(np.sum(residual(p)**2)), p0, method="SLSQP",
                              bounds=list(zip(lo, hi)), constraints=[dict(type="ineq", fun=lambda p: raw_v2(a, p, model))],
                              options={"maxiter": 2000, "ftol": 1e-10})
            v2 = raw_v2(a, result.x, model)
            if result.success and np.isfinite(v2).all() and np.min(v2) >= 0:
                candidates.append((float(np.sum(residual(result.x)**2)), result.x, result, "constrained SLSQP"))
    if not candidates:
        return dict(p=np.full(2, np.nan), success=False, chi2=float("nan"), optimizer="failed",
                    n_starts=len(attempts), n_converged=sum(z.success for z in attempts), n_valid=0,
                    min_total_v2=float("nan"), bound_hit=False, nfev=sum(z.nfev for z in attempts), optimality=float("nan"))
    cost, p, result, method = min(candidates, key=lambda z: z[0])
    return dict(p=p, success=True, chi2=cost, optimizer=method, n_starts=len(attempts),
                n_converged=sum(z.success for z in attempts), n_valid=len(candidates),
                min_total_v2=float(np.min(raw_v2(a, p, model))),
                bound_hit=bool(np.any(np.minimum(p-lo, hi-p) < 1e-4)),
                nfev=int(result.nfev), optimality=float(getattr(result, "optimality", np.nan)))


def fit_record(galaxy: str, a: np.ndarray, model: str, result: dict, scope: str) -> dict:
    p = result["p"]
    pred = prediction(a, p, model) if result["success"] else np.full(len(a), np.nan)
    n = len(a)
    record = {k: v for k, v in result.items() if k != "p"}
    record.update(galaxy=galaxy, model=model, scope=scope, n=n, k=K,
                  p0_log10=float(p[0]), p1_log10=float(p[1]),
                  p0_physical=float(10.**p[0]), p1_physical=float(10.**p[1]),
                  rmse=float(np.sqrt(np.mean((pred-a[:, 1])**2))),
                  chi2_reduced=result["chi2"]/(n-K) if n > K else np.nan,
                  bic=result["chi2"] + K*np.log(n) + float(np.log(2*np.pi*a[:, 2]**2).sum()))
    return record


def outer_split(n: int) -> tuple[int, str]:
    n_train = max(5, int(np.floor(.7*n)))
    return n_train, "eligible" if n_train <= n-2 else "fewer than five training and two test points"


def bootstrap_pair(delta: np.ndarray, rng: np.random.Generator) -> dict:
    samples = delta[rng.integers(0, len(delta), size=(5000, len(delta)))]
    mean_ci = np.quantile(samples.mean(axis=1), [.025, .975])
    median_ci = np.quantile(np.median(samples, axis=1), [.025, .975])
    return dict(n=int(len(delta)), mean_difference=float(np.mean(delta)),
                mean_ci95=[float(x) for x in mean_ci], median_difference=float(np.median(delta)),
                median_ci95=[float(x) for x in median_ci], fluid_better_count=int((delta < -1e-6).sum()),
                comparator_better_count=int((delta > 1e-6).sum()), ties=int((np.abs(delta) <= 1e-6).sum()),
                numerical_tie_tolerance=1e-6)


def numeric_checks() -> dict:
    results = {}
    # Tiny-radius limits protect the cancellation-sensitive halo expressions.
    r = np.array([1e-7, 1e-6, 1e-5])
    p = np.array([7., 1.])
    expected_nfw = 2*np.pi*G*10.**p[0]*10.**p[1]*r
    expected_core = 4*np.pi*G*10.**p[0]*r**2/3
    results["nfw_small_radius_relative_error"] = float(np.max(np.abs(halo_v2(r, p, MODELS[1])/expected_nfw-1)))
    results["burkert_small_radius_relative_error"] = float(np.max(np.abs(halo_v2(r, p, MODELS[2])/expected_core-1)))
    assert results["nfw_small_radius_relative_error"] < 2e-6
    assert results["burkert_small_radius_relative_error"] < 2e-6
    toy = np.array([[1., 40., 1., -30., 20., 10., 0., 0.]])
    assert np.isclose(baryonic_v2(toy)[0], -630.)
    tiny = np.array([-6., -3.])
    assert np.isnan(prediction(toy, tiny, MODELS[1])).all()
    results["signed_gas_preserved"] = True
    results["negative_total_force_rejected"] = True
    for n in range(4, 200):
        split, status = outer_split(n)
        if status == "eligible":
            assert split >= 5 and n-split >= 2 and split > K
    results["outer_split_minimum_checks"] = True
    # Exact halo-only distance scale degeneracy: x and sigma are invariant.
    rows = []
    radii = np.geomspace(.01, 100., 100)
    for logrho, sigma in ((5., 30.), (7., 100.), (9., 250.)):
        base = fluid_circular_speed(radii, logrho, math.log10(sigma))
        for scale in (.25, .5, 2., 4.):
            scaled = fluid_circular_speed(radii*scale, logrho-2*math.log10(scale), math.log10(sigma))
            relative = float(np.max(np.abs(scaled-base)/np.maximum(np.abs(base), 1e-20)))
            rows.append(dict(log10_rho0=logrho, sigma_kms=sigma, distance_scale=scale, max_relative_speed_difference=relative))
    pd.DataFrame(rows).to_csv(OUT/"halo_distance_degeneracy.csv", index=False)
    results["halo_only_distance_degeneracy_max_relative_error"] = max(row["max_relative_speed_difference"] for row in rows)
    assert results["halo_only_distance_degeneracy_max_relative_error"] < 1e-9
    # Direct target-leakage guard: changing only heldout velocities cannot alter
    # training inputs, starts, or the trained parameters.
    a = load_curve(ROOT/"data/NGC2403_rotmod.dat")
    example_halo2 = np.asarray(fluid_circular_speed(a[:, 0], 7., 2.))**2
    reference_total = np.sqrt(example_halo2+baryonic_v2(a))
    baryon_scale_rows = []
    for scale in (.25, .5, 1., 2., 4.):
        scaled_halo2 = np.asarray(fluid_circular_speed(a[:, 0]*scale, 7.-2*math.log10(scale), 2.))**2
        # At fixed observed angular photometry and inclination, gas and stellar
        # v^2 templates scale as distance. Fixed M/L breaks the halo-only rule.
        total = np.sqrt(scaled_halo2+scale*baryonic_v2(a))
        for j, radius in enumerate(a[:, 0]):
            baryon_scale_rows.append(dict(galaxy="NGC2403", distance_scale=scale, reference_radius_kpc=float(radius),
                                          reference_speed_kms=float(reference_total[j]), scaled_speed_kms=float(total[j]),
                                          fixed_upsilon=True))
    pd.DataFrame(baryon_scale_rows).to_csv(OUT/"fixed_upsilon_baryon_distance_scaling.csv", index=False)
    assert np.max(np.abs(total/reference_total-1.)) > .01
    results["fixed_upsilon_baryons_break_halo_only_distance_invariance"] = True
    split, _ = outer_split(len(a))
    changed = a.copy()
    changed[split:, 1] += 10000.
    original_fit = fit_curve(a[:split], MODELS[0])
    changed_fit = fit_curve(changed[:split], MODELS[0])
    assert original_fit["success"] and changed_fit["success"]
    error = float(np.max(np.abs(original_fit["p"]-changed_fit["p"])))
    assert error == 0.
    results["heldout_target_perturbation_parameter_difference"] = error
    recovery = {}
    for model, true_p in ((MODELS[0], [7., math.log10(80.)]), (MODELS[1], [7., 1.]),
                          (MODELS[2], [7., 1.]), (MODELS[3], [math.log10(120.), math.log10(3.)])):
        synthetic = np.zeros((60, 8))
        synthetic[:, 0] = np.geomspace(.02, 100., 60)
        synthetic[:, 2] = 1.
        synthetic[:, 1] = prediction(synthetic, np.array(true_p), model)
        recovered = fit_curve(synthetic, model)
        assert recovered["success"]
        parameter_error = float(np.max(np.abs(recovered["p"]-true_p)))
        assert parameter_error < 1e-5
        recovery[model] = parameter_error
    results["noiseless_synthetic_max_log10_parameter_error"] = recovery
    return results


def make_plots(full: pd.DataFrame, held: pd.DataFrame, predictions: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#386cb0", "#f28e2b", "#59a14f", "#9c6ade")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    data = [held.loc[(held.model == m) & held.test_prediction_valid, "test_rmse"].to_numpy() for m in MODELS]
    boxes = axes[0].boxplot(data, tick_labels=["Fluid", "NFW", "Burkert", "Exponential"], showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(.65)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Outer holdout RMSE (km/s)")
    axes[0].set_title("Fixed models; training uses only inner velocities")
    wide = held.pivot(index="galaxy", columns="model", values="test_rmse")
    for model, color in zip(MODELS[1:], colors[1:]):
        axes[1].scatter(wide[model], wide[MODELS[0]], s=14, alpha=.55, color=color, label=model.replace(" + baryons", ""))
    limits = [.5, max(500, float(np.nanmax(wide.to_numpy()))*1.05)]
    axes[1].plot(limits, limits, "k--", lw=1)
    axes[1].set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                xlabel="Comparator holdout RMSE (km/s)", ylabel="Fluid holdout RMSE (km/s)")
    axes[1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT/"holdout_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # Prechosen illustrative galaxies; these panels are not a winner selection.
    for ax, galaxy in zip(axes.ravel(), ("NGC2403", "DDO154", "NGC3198", "UGC01281")):
        local = predictions[predictions.galaxy == galaxy]
        obs = local.drop_duplicates("radius_kpc")
        if obs.empty:
            continue
        ax.errorbar(obs.radius_kpc, obs.v_observed, yerr=obs.error_kms, fmt=".", ms=4, color="black", alpha=.65, label="Observed")
        train_max = float(obs.loc[obs.partition == "train", "radius_kpc"].max())
        ax.axvspan(train_max, float(obs.radius_kpc.max()), color="#ddd", alpha=.55, label="Held out")
        for model, color in zip(MODELS, colors):
            line = local[local.model == model]
            ax.plot(line.radius_kpc, line.v_predicted, color=color, lw=1.6, label=model.replace(" + baryons", ""))
        ax.set(title=galaxy, xlabel="Radius (kpc)", ylabel="Speed (km/s)")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT/"outer_prediction_examples.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only the first N galaxies; uses separate output directory.")
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--wide-bounds", action="store_true", help="Sensitivity run with much wider bounds, saved separately.")
    args = parser.parse_args()
    global OUT, WIDE_BOUNDS
    WIDE_BOUNDS = args.wide_bounds
    if args.wide_bounds:
        OUT = OUT / "wide_bounds"
    if args.limit:
        OUT = OUT / f"smoke_{args.limit}"
    OUT.mkdir(parents=True, exist_ok=True)
    checks = numeric_checks()
    (OUT/"numerical_checks.json").write_text(json.dumps(checks, indent=2))
    if args.checks_only:
        print(json.dumps(checks, indent=2)); return
    started = time.monotonic()
    full_rows, held_rows, pred_rows, split_rows = [], [], [], []
    paths = sorted((ROOT/"data").glob("*_rotmod.dat"))[:args.limit]
    for index, path in enumerate(paths, start=1):
        a = load_curve(path)
        galaxy = path.stem.removesuffix("_rotmod")
        n_train, status = outer_split(len(a))
        split_rows.append(dict(galaxy=galaxy, n=len(a), n_train=n_train if status == "eligible" else 0,
                               n_test=len(a)-n_train if status == "eligible" else 0, holdout_status=status,
                               negative_gas_points=int(np.sum(a[:, 3] < 0)),
                               source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for model in MODELS:
            # Full fits and training fits never seed one another.
            full_result = fit_curve(a, model)
            full_rows.append(fit_record(galaxy, a, model, full_result, "all radii"))
            if status != "eligible":
                continue
            training, testing = a[:n_train], a[n_train:]
            trained = fit_curve(training, model)
            record = fit_record(galaxy, training, model, trained, "inner training")
            test_pred = prediction(testing, trained["p"], model) if trained["success"] else np.full(len(testing), np.nan)
            valid = bool(trained["success"] and np.isfinite(test_pred).all())
            record.update(n_test=len(testing), test_prediction_valid=valid,
                          train_radius_max_kpc=float(training[-1, 0]), test_radius_min_kpc=float(testing[0, 0]),
                          test_rmse=float(np.sqrt(np.mean((test_pred-testing[:, 1])**2))),
                          test_chi2_per_point=float(np.mean(((test_pred-testing[:, 1])/testing[:, 2])**2)),
                          test_mean_error_kms=float(np.mean(test_pred-testing[:, 1])))
            held_rows.append(record)
            all_pred = prediction(a, trained["p"], model) if trained["success"] else np.full(len(a), np.nan)
            for j in range(len(a)):
                pred_rows.append(dict(galaxy=galaxy, model=model, partition="train" if j < n_train else "test",
                                      radius_kpc=float(a[j, 0]), v_observed=float(a[j, 1]), error_kms=float(a[j, 2]), v_predicted=float(all_pred[j])))
        if index % 10 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} galaxies; elapsed {time.monotonic()-started:.1f}s", flush=True)
            pd.DataFrame(full_rows).to_csv(OUT/"full_fits_partial.csv", index=False)
            pd.DataFrame(held_rows).to_csv(OUT/"holdout_partial.csv", index=False)
    full, held = pd.DataFrame(full_rows), pd.DataFrame(held_rows)
    predictions, splits = pd.DataFrame(pred_rows), pd.DataFrame(split_rows)
    full.to_csv(OUT/"full_fits.csv", index=False)
    held.to_csv(OUT/"holdout_fits.csv", index=False)
    predictions.to_csv(OUT/"outer_predictions.csv", index=False)
    splits.to_csv(OUT/"galaxy_splits.csv", index=False)
    for partial in OUT.glob("*_partial.csv"):
        partial.unlink()
    rng = np.random.default_rng(20261023)
    comparisons = {}
    for score in ("test_rmse", "test_chi2_per_point"):
        wide = held[held.test_prediction_valid].pivot(index="galaxy", columns="model", values=score)
        comparisons[score] = {}
        for comparator in MODELS[1:]:
            paired = wide[[MODELS[0], comparator]].dropna()
            comparisons[score][comparator] = bootstrap_pair((paired[MODELS[0]]-paired[comparator]).to_numpy(), rng)
    valid_full = full[full.success]
    bic_wide = valid_full.pivot(index="galaxy", columns="model", values="bic")
    bic_comparisons = {}
    for comparator in MODELS[1:]:
        pair = bic_wide[[MODELS[0], comparator]].dropna()
        delta = pair[MODELS[0]] - pair[comparator]
        bic_comparisons[comparator] = dict(n=len(delta), fluid_lower_bic=int((delta < -1e-6).sum()),
                                           comparator_lower_bic=int((delta > 1e-6).sum()),
                                           numerical_ties=int((np.abs(delta) <= 1e-6).sum()), numerical_tie_tolerance=1e-6,
                                           fluid_strong_delta_less_minus10=int((delta < -10).sum()),
                                           comparator_strong_delta_above10=int((delta > 10).sum()),
                                           median_fluid_minus_comparator_bic=float(delta.median()))
    summary = dict(
        configuration=dict(models=list(MODELS), parameters_per_model=2, fixed_mass_to_light_disk=.5,
                           fixed_mass_to_light_bulge=.7, errors="Quoted independent point errors only; no added error floor",
                           signed_gas="Vgas*abs(Vgas)", negative_total_force="Reject physical predictions; explicit constraint fallback during fitting",
                           full_data_bic="chi2 + k log(n) + sum(log(2*pi*error^2)); same data and k for every model",
                           outer_split="Inner max(5,floor(0.7*n)) observations; at least 2 outer test observations",
                           starts="Fluid: convex amplitude profiling + 420-point log-core-radius grid with every local minimum refined; NFW/Burkert:5 deterministic starts; exponential:profiled amplitude+180-point radius grid; training data only, no full-data fitted seeds",
                           holdout_normalization="chi2/n_test; no parameters fit on heldout points",
                           bootstrap="5000 paired galaxy resamples, seed20261023; descriptive percentile95% CI",
                           bounds={m: [bounds_for(m)[0].tolist(), bounds_for(m)[1].tolist()] for m in MODELS},
                           parameter_units={MODELS[0]: ["rho0 Msun/kpc^3", "sigma km/s"], MODELS[1]: ["rho_s Msun/kpc^3", "r_s kpc"],
                                            MODELS[2]: ["rho0 Msun/kpc^3", "r_s kpc"], MODELS[3]: ["asymptotic speed km/s", "turnover radius kpc"]}),
        data=dict(galaxies=len(splits), points=int(splits.n.sum()), heldout_galaxies=int((splits.holdout_status == "eligible").sum()),
                  excluded_holdout_galaxies=splits.loc[splits.holdout_status != "eligible", "galaxy"].tolist(),
                  negative_gas_points=int(splits.negative_gas_points.sum())),
        full_fit_failures=full.groupby("model").success.apply(lambda s: int((~s).sum())).to_dict(),
        training_fit_failures=held.groupby("model").success.apply(lambda s: int((~s).sum())).to_dict(),
        invalid_test_predictions=held.groupby("model").test_prediction_valid.apply(lambda s: int((~s).sum())).to_dict(),
        full_fit_bound_hits=full.groupby("model").bound_hit.sum().astype(int).to_dict(),
        training_fit_bound_hits=held.groupby("model").bound_hit.sum().astype(int).to_dict(),
        median_full_reduced_chi2=full.groupby("model").chi2_reduced.median().to_dict(),
        median_holdout_rmse=held.groupby("model").test_rmse.median().to_dict(),
        median_holdout_chi2_per_point=held.groupby("model").test_chi2_per_point.median().to_dict(),
        paired_holdout_comparisons=comparisons, full_bic_comparisons=bic_comparisons,
        numerical_checks=checks, elapsed_seconds=time.monotonic()-started,
        interpretation_limits=["The isothermal fluid fit tests one hydrostatic halo closure, not the original drag formula or viscosity.",
                               "Uncontracted baryon addition neglects the halo response; this is not a self-consistent baryon-coupled fluid solution.",
                               "SPARC rotation speeds and baryonic templates use adopted distances and inclinations; these fits do not infer distance.",
                               "Outer radial testing is within-galaxy extrapolation with known baryonic templates; not prediction for unseen galaxies or images.",
                               "Quoted errors omit many systematics and correlations; large chi2 is not automatically evidence for one dark-matter microphysics.",
                               "Bound hits and weakly identified parameters invalidate simple interior-Gaussian BIC assumptions; BIC here is a descriptive fit diagnostic, not a Bayes factor.",
                               "No postselected best-family holdout score is presented as an unbiased predictor."])
    metadata_path = ROOT/"executed_research/images/sparc_metadata.csv"
    if metadata_path.exists():
        # Explicitly posthoc: downloaded after the primary protocol was chosen.
        # No model is refit and the all-galaxy primary experiment is retained.
        metadata = pd.read_csv(metadata_path)
        selected_names = set(metadata.loc[(metadata.quality_flag <= 2) & (metadata.inclination_deg >= 30), "galaxy"])
        selected = held[held.galaxy.isin(selected_names)]
        sensitivity = dict(selection="Posthoc quality_flag<=2 and inclination_deg>=30; no refitting or favorable-cut selection",
                           metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                           full_galaxies=int(full.loc[full.galaxy.isin(selected_names), "galaxy"].nunique()),
                           heldout_galaxies=int(selected.galaxy.nunique()),
                           median_holdout_rmse=selected.groupby("model").test_rmse.median().to_dict(),
                           median_holdout_chi2_per_point=selected.groupby("model").test_chi2_per_point.median().to_dict(),
                           paired_holdout_comparisons={})
        subset_rng = np.random.default_rng(20261024)
        for score in ("test_rmse", "test_chi2_per_point"):
            wide = selected[selected.test_prediction_valid].pivot(index="galaxy", columns="model", values=score)
            sensitivity["paired_holdout_comparisons"][score] = {}
            for comparator in MODELS[1:]:
                pair = wide[[MODELS[0], comparator]].dropna()
                sensitivity["paired_holdout_comparisons"][score][comparator] = bootstrap_pair((pair[MODELS[0]]-pair[comparator]).to_numpy(), subset_rng)
        summary["posthoc_quality_sensitivity"] = sensitivity
        (OUT/"posthoc_quality_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2))
    from experiments.write_dynamics_interpretation import write_interpretation
    write_interpretation(OUT)
    make_plots(full, held, predictions)
    print(json.dumps({"data": summary["data"], "median_holdout_rmse": summary["median_holdout_rmse"],
                      "failures": summary["full_fit_failures"], "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
