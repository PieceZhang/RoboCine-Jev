"""Crash-safe JSON persistence: atomic write + one-time backup + load.

Replaces 7+ ad-hoc copies of `tmp + os.replace` and `json.dump` across the
original scripts (including upgrading final_annotation's non-atomic write).
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

_backed_up = set()


def atomic_write_json(path, obj, indent=2, ensure_ascii=False):
    """Write obj to path atomically: write path+'.tmp' then os.replace."""
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    os.replace(tmp, path)


def load_json(path, default=None):
    """Load JSON from path; return default (or {} / [] if None) on missing/error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} if default is None else default
    except Exception:
        return {} if default is None else default


def backup_once(path, tag="bak"):
    """Copy path -> path + '.<tag>-<timestamp>' the first time it is seen.
    Idempotent per path within a process."""
    path = str(path)
    if path in _backed_up or not Path(path).exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = f"{path}.{tag}-{ts}"
    shutil.copy2(path, bak)
    _backed_up.add(path)
    return bak
