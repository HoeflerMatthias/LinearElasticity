import hashlib
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    """Load a YAML or JSON config file into a plain dict."""
    p = Path(path)
    with p.open("r") as f:
        if p.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        else:
            return json.load(f)


def compute_config_hash(
    params: Dict[str, Any],
    seed: Optional[int] = None,
    exclude_keys: Optional[List[str]] = None,
) -> str:
    """Compute a stable MD5 hash of the config (optionally with seed)."""
    def _filter(d):
        return {
            k: _filter(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if not exclude_keys or k not in exclude_keys
        }

    raw = json.dumps(_filter(params), sort_keys=True, default=str)
    if seed is not None:
        raw += str(seed)
    return hashlib.md5(raw.encode()).hexdigest()


def flatten_params(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, key))
        elif isinstance(v, (list, tuple)):
            flat[key] = str(v)
        else:
            flat[key] = v
    return flat
