"""Reproduce the archived read-only SDSS query, or verify the cached catalog."""
from pathlib import Path
import argparse
import hashlib
import json
from io import StringIO
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "executed_research/images/sdss_resolved_sample.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    record = json.loads((DEST.parent / "sdss_query_provenance.json").read_text())
    if not DEST.exists() or args.refresh:
        if args.offline:
            raise FileNotFoundError("Cached SDSS catalog missing; offline mode never downloads")
        if urlparse(record["url"]).hostname != "skyserver.sdss.org":
            raise ValueError("The acquisition record must use the official SDSS service")
        response = requests.get(record["url"], timeout=(10, 90))
        response.raise_for_status()
        candidate = pd.read_csv(StringIO(response.text), comment="#", dtype={"galaxy_id":str})
        if len(candidate) != 500 or candidate.galaxy_id.duplicated().any():
            raise ValueError("Response does not match the declared 500 unique-galaxy catalog")
        DEST.write_bytes(response.content)
    print(json.dumps(dict(path=str(DEST), bytes=DEST.stat().st_size,
                          sha256=hashlib.sha256(DEST.read_bytes()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
