"""Small, dependency-light helpers shared by training, evaluation and tests."""
import hashlib
import json
import random
import platform
from importlib import metadata
from pathlib import Path
import numpy as np


def seed_case(seed, case):
    return (int(seed) + int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)) % (2**32)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng_state():
    """Capture RNGs so validation sampling cannot perturb later training."""
    import torch
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def write_manifest(path, opt, **extra):
    files = {}
    for key in ('ymlpath', 'datasetfile', 'valid_datasetfile'):
        p = Path(getattr(opt, key, ''))
        if p.is_file():
            files[key] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    versions = {'python': platform.python_version()}
    for package in ('torch', 'numpy', 'scipy', 'scikit-image', 'h5py'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    options = dict(opt) if hasattr(opt, 'keys') else vars(opt)
    payload = {'protocol': 'revision_v2', 'options': options, 'input_files': files,
               'software_versions': versions, **extra}
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
