"""GT-writeback factory.

The three GT servers each snapshot the model's original prediction on first
human correction, but under different keys:
  depth viewer -> 'predicted' / 'predicted_model'
  VLM audit    -> 'vlm_predicted' / 'vlm_model'
  mined clean  -> 'depth_predicted' / 'depth_model'
This factory unifies the logic behind parameters so each tool keeps its OWN
existing key name (do not rename — downstream readers depend on them).
"""
import threading
from datetime import datetime

from .schema import DIRS
from .atomic_io import atomic_write_json, backup_once


def make_save_gt(json_path, snapshot_key, snapshot_model_key, lock=None):
    """Return save_gt(data, key, flags) -> (ok, info).

    On first correction: snapshot the model's original 5 bools under
    snapshot_key (and model id under snapshot_model_key), then overwrite the
    top-level flags with human GT, stamp corrected metadata, atomic-write."""
    lock = lock or threading.Lock()

    def save_gt(data, key, flags):
        with lock:
            if key not in data:
                return False, "relative_path 不在 JSON 中"
            entry = data[key]
            backup_once(json_path)
            if snapshot_key not in entry:
                entry[snapshot_key] = {d: bool(entry.get(d, False)) for d in DIRS}
                entry[snapshot_model_key] = entry.get("model", "")
            for d in DIRS:
                entry[d] = bool(flags.get(d, False))
            entry["corrected"] = True
            entry["corrected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry["corrected_by"] = "manual_gt"
            atomic_write_json(json_path, data)
            return True, {d: bool(flags.get(d, False)) for d in DIRS}

    return save_gt
