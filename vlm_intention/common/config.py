"""Config loader: YAML + ${paths.x} interpolation + nested attribute access.

The shipped config.yaml carries the CURRENT (pre-refactor) values as defaults,
so a no-op config reproduces today's behavior. A user config (--config) is
deep-merged over the shipped defaults.
"""
import os
import re
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from e

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../VLMIntention_ws
_SHIPPED = _REPO_ROOT / "config.yaml"
_INTERP = re.compile(r"\$\{([^}]+)\}")


class DotDict(dict):
    """dict with attribute access for nested config keys."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return DotDict(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _deep_merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _interpolate(obj, root):
    """Recursively replace ${a.b.c} with root[a][b][c] (string-valued leaves)."""
    if isinstance(obj, dict):
        return {k: _interpolate(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, root) for v in obj]
    if isinstance(obj, str):
        def sub(m):
            keys = m.group(1).split(".")
            cur = root
            for kk in keys:
                cur = cur[kk] if isinstance(cur, dict) else getattr(cur, kk)
            return str(cur)
        # repeat until stable (allows chained refs)
        prev = None
        s = obj
        while prev != s:
            prev = s
            s = _INTERP.sub(sub, s)
        return s
    return obj


def load_config(path=None):
    """Load shipped defaults, deep-merge the user config (if any), interpolate
    ${...}, and resolve relative paths against the repo root. Returns a DotDict."""
    base = yaml.safe_load(_SHIPPED.read_text(encoding="utf-8")) if _SHIPPED.exists() else {}
    if path and Path(path).exists():
        over = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _deep_merge(base, over)
    merged = _interpolate(base, base)
    return DotDict(merged)


def repo_root():
    """Absolute path to the repo root (VLMIntention_ws/)."""
    return _REPO_ROOT
