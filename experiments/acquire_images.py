"""Acquire a modest public FITS pilot; retain hashes, WCS and failures.

python experiments/acquire_images.py            # download missing images
python experiments/acquire_images.py --offline  # inventory cached files only

Query centers are approximate named-galaxy centers, not new astrometry.
All cutouts are resampled survey pixels; they do not include precision PSF,
mask, or variance products. No absolute photometry or distance accuracy is
claimed from this small, deliberately selected morphology pilot.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from astropy.io import fits
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "executed_research/images"
# Coordinates are used only to center fixed angular fields; never to set kpc
# crops or select a distance-derived physical resolution.
TARGETS = [
    ("M51", 202.4696, 47.1952, 2.0),
    ("M101", 210.8023, 54.3489, 4.0),
    ("NGC0628", 24.1741, 15.7835, 2.0),
    ("NGC2403", 114.2142, 65.6026, 4.0),
    ("NGC3198", 154.9788, 45.5496, 2.0),
    ("NGC5055", 198.9555, 42.0293, 2.5),
    ("NGC6946", 308.7180, 60.1539, 2.5),
    ("NGC3938", 178.2062, 44.1207, 1.5),
]


def target_specs():
    for name, ra, dec, scale in TARGETS:
        for band in (("g", "r", "z") if name == "M51" else ("r",)):
            params = dict(ra=ra, dec=dec, layer="ls-dr9", pixscale=scale, size=256, bands=band)
            yield dict(id=name + "_" + band, path=str((OUT / (name + "_" + band + ".fits")).relative_to(ROOT)),
                       ra_deg=ra, dec_deg=dec, pixel_scale_arcsec=scale, band=band,
                       source_url="https://www.legacysurvey.org/viewer/fits-cutout?" + urlencode(params),
                       coordinate_provenance="Approximate named target query center; inspect centering",
                       survey_release="Legacy Surveys DR9", selection="Convenience morphology pilot",
                       image_products_missing=["PSF", "inverse variance", "survey mask", "verified photometric zero point"])


def inventory_one(spec, download=False):
    result = spec.copy()
    path = ROOT / spec["path"]
    try:
        if not path.exists():
            if not download:
                raise FileNotFoundError("Not cached; offline inventory does not download")
            response = requests.get(spec["source_url"], timeout=(10, 60))
            response.raise_for_status()
            # Validate response before treating it as a scientific input.
            if not response.content.startswith(b"SIMPLE"):
                raise ValueError("Response is not a primary FITS image")
            path.write_bytes(response.content)
        with fits.open(path) as hdus:
            data = hdus[0].data
            if data is None or data.ndim != 2:
                raise ValueError("Expected a 2D science cutout")
            if np.count_nonzero(np.isfinite(data) & (data != 0)) < data.size * .2:
                raise ValueError("Cutout is mostly empty/missing survey coverage")
            result.update(shape=list(data.shape), finite_fraction=float(np.isfinite(data).mean()),
                          fits_bunit=hdus[0].header.get("BUNIT"),
                          fits_survey=hdus[0].header.get("SURVEY"),
                          fits_release=hdus[0].header.get("VERSION"), hdu=0)
        result.update(status="available", bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception as error:
        result.update(status="failed", error=str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = list(target_specs())
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda spec: inventory_one(spec, not args.offline), specs))
    manifest = dict(acquired_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_documentation="https://www.legacysurvey.org/viewer/urls",
                    purpose="Exploratory angular morphology; not an independent distance dataset",
                    images=[r for r in results if r["status"] == "available"],
                    failures=[r for r in results if r["status"] != "available"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "targets.json").write_text(json.dumps(specs, indent=2) + "\n")
    print(json.dumps(dict(available=len(manifest["images"]), failed=len(manifest["failures"]),
                         manifest=str(OUT / "manifest.json")), indent=2))


if __name__ == "__main__":
    main()
