"""Package a private Factor source/evidence snapshot without environments or base weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TOP = {'.git', '.venv', '.venv-local', 'runs', 'build', 'outputs'}
EXCLUDED_LOCAL = {'base_model', 'cache', 'models', 'checkpoints'}
MANIFEST = 'RELEASE_MANIFEST.json'


def included(path):
    parts = path.relative_to(ROOT).parts
    if parts[0] in EXCLUDED_TOP or parts[0].startswith('.venv-'):
        return False
    if any(p in {'__pycache__', '.pytest_cache', '.DS_Store'} or p.endswith('.egg-info') for p in parts):
        return False
    if len(parts) > 2 and parts[:2] == ('experiments', 'local_model') and parts[2] in EXCLUDED_LOCAL:
        return False
    if path.name == MANIFEST or path.suffix in {'.pyc', '.pyo', '.gguf'}:
        return False
    if path.name.startswith('.env') and path.name != '.env.example':
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Write the bundle outside the source tree')
    checksum = output.with_suffix(output.suffix + '.sha256')
    if output.exists() or checksum.exists():
        raise FileExistsError('Preserve the existing release and choose a new output name')
    files = []
    for path in sorted(ROOT.rglob('*')):
        if not included(path):
            continue
        if path.is_symlink():
            raise ValueError(f'Refuse symlink: {path}')
        if not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) > 50_000_000:
            raise ValueError(f'Unexpected large release file: {path}')
        files.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()})
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {'schema_version': 'factor-source-release/1', 'version': '0.3.0',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'base_git_commit': commit,
                'kind': 'private local uncommitted source and evidence snapshot',
                'scope': 'synthetic development; no frontier, real Slurm, human or real-data validation',
                'excludes': ['environments', 'large base weights', 'caches', 'Git internals', 'top-level runs'],
                'retains': ['reports/**/runs', 'trained small adapter', 'source snapshots', 'tested wheel'],
                'manifest_self_hash': 'excluded; use the external ZIP checksum', 'files': files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for record in files:
            data = (ROOT / record['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f"File changed during packaging: {record['path']}")
            archive.writestr('factor/' + record['path'], data)
        archive.writestr('factor/' + MANIFEST, json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(output) as archive:
        expected = {'factor/' + f['path'] for f in files} | {'factor/' + MANIFEST}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError('Unexpected archive entries')
        for record in files:
            data = archive.read('factor/' + record['path'])
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('Archive verification failed')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum.write_text(f'{digest}  {output.name}\n')
    print(json.dumps({'archive': str(output), 'files': len(files), 'bytes': output.stat().st_size,
                      'sha256': digest, 'manifest_verified': True}, indent=2))


if __name__ == '__main__':
    main()
