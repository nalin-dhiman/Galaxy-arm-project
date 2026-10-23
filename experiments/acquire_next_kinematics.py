#!/usr/bin/env python3
"""Acquire only the six preregistered public MaNGA MAPS files."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone

import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'executed_next/kinematics'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    directory = OUT / 'data'
    directory.mkdir(exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old_files = {r['id']: r for r in old.get('files', [])}
    files, failures = [], []
    for target in protocol['targets']:
        path = directory / target['url'].split('/')[-1]
        try:
            if not path.exists():
                if not args.download:
                    raise FileNotFoundError('Not cached; use --download')
                for attempt in range(3):
                    response = requests.get(target['url'], timeout=(20, 120))
                    if response.status_code in (429, 503) and attempt < 2:
                        time.sleep(min(60, max(5, float(response.headers.get('Retry-After', 15)))))
                        continue
                    response.raise_for_status()
                    temporary = path.with_suffix(path.suffix + '.part')
                    temporary.write_bytes(response.content)
                    with fits.open(temporary) as hdus:
                        assert str(hdus[0].header['PLATEIFU']) == target['plateifu']
                        for name in ('EMLINE_GVEL', 'EMLINE_GVEL_IVAR', 'EMLINE_GVEL_MASK',
                                     'EMLINE_GFLUX', 'EMLINE_GFLUX_IVAR', 'EMLINE_GFLUX_MASK'):
                            assert hdus[name].data.ndim == 3
                    temporary.replace(path)
                    break
                time.sleep(1)
            digest = sha(path)
            if target['id'] in old_files and digest != old_files[target['id']]['sha256']:
                raise ValueError('Previously recorded file checksum changed')
            with fits.open(path) as hdus:
                row = dict(id=target['id'], plateifu=target['plateifu'], url=target['url'],
                           path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                           sha256=digest, dapqual=int(hdus[0].header.get('DAPQUAL', 0)))
            files.append(row)
            print('Cached', target['id'], path.stat().st_size, flush=True)
        except Exception as exc:
            failures.append(dict(id=target['id'], error=f'{type(exc).__name__}: {exc}'))
            print('Failed', target['id'], str(exc), flush=True)
    result = dict(status='passed' if len(files) == 6 and not failures else 'incomplete',
                  requested=6, completed=len(files), failures=failures, files=files,
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(OUT/'protocol.json'))
    manifest_path.write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
