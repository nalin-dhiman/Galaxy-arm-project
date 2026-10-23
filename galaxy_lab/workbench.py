"""Local multiband analysis report with explicit input and model scope."""
from pathlib import Path
from html import escape
import hashlib
import json
import os
import joblib
import numpy as np
import pandas as pd
from astropy.io import fits
from .filtering import DiskGeometry, display_transform
from .native import read_native_grz, annotation_center_and_stars, reference_center_sky
from .redshift import observable_frame, cosmological_distances
from .segmentation import predict_arm_regions


def analyze(args):
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be new or empty")
    if (args.center_ra is None) != (args.center_dec is None):
        raise ValueError("Supply both sky center coordinates")
    target_sky = None
    if args.reference_marks:
        if args.center_ra is not None:
            raise ValueError("Choose reference marks or explicit sky coordinates")
        target_sky = reference_center_sky(args.reference_marks)
    elif args.center_ra is not None:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        target_sky = SkyCoord(args.center_ra*u.deg, args.center_dec*u.deg)
    images, source = read_native_grz(args.image, args.psf, declared_unit=args.units,
        mask_path=args.foreground_mask, target_sky=target_sky)
    reference = images["r"]
    if (args.center_x is None) != (args.center_y is None):
        raise ValueError("Supply both center coordinates")
    if args.reference_marks:
        if args.center_x is not None:
            raise ValueError("Choose reference marks or explicit center coordinates")
        center, stars = annotation_center_and_stars(args.reference_marks, reference)
        for image in images.values():
            image.mask = ~image.valid | stars
        geometry_source = "independent center/star inputs; human arm masks never read"
        source["foreground_mask_supplied"] = True
    elif target_sky is not None:
        if args.center_x is not None:
            raise ValueError("Choose sky coordinates or native pixel coordinates")
        center = tuple(float(x) for x in reference.wcs.world_to_pixel(target_sky))
        geometry_source = "user supplied sky center; fixed circular projected annulus"
    elif args.center_x is not None:
        center = (args.center_x, args.center_y)
        geometry_source = "user supplied center; fixed circular projected annulus"
    else:
        center = ((reference.data.shape[1]-1)/2, (reference.data.shape[0]-1)/2)
        geometry_source = "image center default; verify galaxy centering"
    if not np.isfinite(center).all() or not (0 <= center[0] < reference.data.shape[1] and 0 <= center[1] < reference.data.shape[0]):
        raise ValueError("Center must be finite and inside the image")
    geometry = DiskGeometry(center, 0., 0., 3., 23., geometry_source)
    display_result = predict_arm_regions(args.arm_model, images, geometry)
    if display_result["valid"].sum() < 100:
        raise ValueError("Insufficient valid inference pixels")
    result = {"status": "experimental_research_analysis", "source": source,
        "geometry": {"center_xy": list(center), "region_arcsec": [3, 23], "source": geometry_source},
        "arm_model_path": str(args.arm_model),
        "arm_model_sha256": hashlib.sha256(args.arm_model.read_bytes()).hexdigest(),
        "arm_display_descriptors": display_result["descriptors"], "arm_metadata": display_result["metadata"],
        "redshift": None, "fluid_inference": "not identified from pixels; use independent kinematics and physical scale",
        "limitations": ["Experimental nearby strong-spiral model, not validated for arbitrary surveys or galaxy types.",
            "Arm agreement and candidate masks do not validate pitch, physical arm spacing, distance or viscosity.",
            "Training evaluation supplied independent centers and foreground-star masks.",
            "PSF matching is an approximate local Gaussian moment operation; noise covariance is unmodeled.",
            "Input calibration is declared by the caller; display images are not accepted as science passbands."]}
    if not source["foreground_mask_supplied"]:
        result["limitations"].append("No foreground mask supplied: crowded fields can differ materially from validation inputs.")
    if args.redshift_model:
        if not args.catalog or not args.distance_arm_model:
            raise ValueError("Redshift inference requires --catalog and the matching --distance-arm-model")
        redshift_model = joblib.load(args.redshift_model)
        required_hash = redshift_model.metadata.get("distance_arm_model_sha256")
        raw_hash = hashlib.sha256(args.distance_arm_model.read_bytes()).hexdigest()
        if required_hash != raw_hash:
            raise ValueError("Redshift model requires its exact matching raw arm-feature model")
        catalog = pd.read_csv(args.catalog)
        if args.galaxy_id:
            if "id" not in catalog:
                raise ValueError("Multi-object photometry needs an id column")
            catalog = catalog.loc[catalog.id.astype(str) == args.galaxy_id]
        if len(catalog) != 1:
            raise ValueError("Choose exactly one photometry row using --galaxy-id")
        catalog = catalog.reset_index(drop=True)
        raw = predict_arm_regions(args.distance_arm_model, images, geometry)
        descriptors = pd.DataFrame([raw["descriptors"]])
        observed = observable_frame(catalog, descriptors, pd.DataFrame([source["quality_features"]]))
        observed["apparent_arm_features_available"] = 1.
        prediction = redshift_model.predict(observed)
        result["redshift"] = {"prediction": prediction.iloc[0].to_dict(),
            "family": redshift_model.metadata["family"], "label": redshift_model.metadata["target"],
            "interval_scope": "90% group-max residual calibration for comparable groups, not geometric-distance uncertainty",
            "descriptor_scope": "raw TRAIN-only arm scores, threshold0.5; separate from calibrated arm display",
            "model_sha256": hashlib.sha256(args.redshift_model.read_bytes()).hexdigest()}
        if args.cosmological_conversion:
            p = prediction.iloc[0]
            try:
                result["conditional_distances"] = cosmological_distances([p.predicted_catalog_redshift],
                    [p.redshift_interval_lower], [p.redshift_interval_upper])
            except ValueError as exc:
                result["conditional_distances"] = {"status": "unavailable", "reason": str(exc)}
    # Validate every requested model and observable before writing any output.
    out.mkdir(parents=True, exist_ok=True)
    header = reference.wcs.to_header()
    header["BUNIT"] = "annotation agreement"
    header["HISTORY"] = "Experimental human-arm-region agreement model; not physical membership"
    fits.writeto(out / "arm_agreement.fits", display_result["probability"], header)
    fits.writeto(out / "arm_candidates.fits", display_result["binary"].astype(np.uint8), reference.wcs.to_header())
    fits.writeto(out / "valid_mask.fits", display_result["valid"].astype(np.uint8), reference.wcs.to_header())
    if args.redshift_model:
        prediction.to_csv(out / "catalog_redshift_prediction.csv", index=False)
        observed.to_csv(out / "observed_predictors.csv", index=False)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/solid_galaxy_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), constrained_layout=True)
    for ax, band in zip(axes[:3], "grz"):
        ax.imshow(display_transform(images[band].data), origin="lower", cmap="gray")
        ax.set_title(band + " (own asinh stretch)")
    score_cmap = plt.get_cmap("magma").copy(); score_cmap.set_bad("#888888")
    binary_cmap = plt.get_cmap("gray").copy(); binary_cmap.set_bad("#888888")
    axes[3].imshow(display_result["probability"], origin="lower", cmap=score_cmap, vmin=0, vmax=1)
    axes[3].set_title("Region agreement score (0–1)\nGray: excluded", fontsize=10)
    axes[4].imshow(np.ma.array(display_result["binary"].astype(float), mask=~display_result["valid"]),
                   origin="lower", cmap=binary_cmap, vmin=0, vmax=1)
    axes[4].set_title("White: candidate; black: other\nGray: excluded; verify candidates", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("Experimental multiband galaxy analysis")
    fig.savefig(out / "analysis.png", dpi=160); plt.close(fig)
    # Convert numpy values without dropping metadata or encoding nonfinite JSON.
    def safe(value):
        if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [safe(v) for v in value]
        if isinstance(value, np.ndarray): return safe(value.tolist())
        if isinstance(value, np.generic): return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    result = safe(result)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    redshift_text = "No photometric-redshift model requested."
    if result["redshift"]:
        p = result["redshift"]["prediction"]
        redshift_text = f'Predicted catalog redshift: {p["predicted_catalog_redshift"]:.4f}; interval [{p["redshift_interval_lower"]:.4f}, {p["redshift_interval_upper"]:.4f}]. This is not a geometric-distance measurement.'
        redshift_text += f' Outside training feature ranges: {p["outside_training_feature_range"]}; missing predictors: {p["missing_predictors"]}.'
    html = '<!doctype html><meta charset="utf-8"><title>Galaxy analysis</title><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}</style>'
    html += '<h1>Experimental galaxy analysis</h1><p>' + escape(redshift_text) + '</p><img src="analysis.png" alt="Science bands, arm score and candidates">'
    html += '<p>Inspect the original bands and mask before interpreting a candidate as an arm.</p><ul>'
    html += ''.join('<li>' + escape(item) + '</li>' for item in result["limitations"]) + '</ul>'
    html += '<p>Native probability/candidate FITS and analysis.json preserve WCS, input hashes and model scope. Fluid parameters and viscosity require additional dynamical information.</p>'
    (out / "analysis.html").write_text(html)
    return {"output_dir": str(out.resolve()), "report": str((out / "analysis.html").resolve()),
            "status": result["status"], "redshift": result["redshift"]}
