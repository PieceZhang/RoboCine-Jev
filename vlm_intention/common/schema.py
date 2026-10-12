"""The 5-direction vocabulary + the 5-bool annotation envelope.

Shared schema across every artifact (depth / VLM / mined / pilot / train):
    {前, 左, 右, 上, 下: bool, image_name, relative_path, timestamp, model[, source]}
"""
import time

DIRS = ["前", "左", "右", "上", "下"]
HORIZ_DIRS = ["前", "左", "右"]
VERT_DIRS = ["上", "下"]


def build_annotation_record(directions, image_name, relative_path, model,
                            source=None, timestamp=None):
    """Assemble a schema-conformant record from a {dir: bool} mapping."""
    rec = {d: bool(directions.get(d, False)) for d in DIRS}
    rec["image_name"] = image_name
    rec["relative_path"] = relative_path
    rec["timestamp"] = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    rec["model"] = model
    if source:
        rec["source"] = source
    return rec


def read_flags(entry):
    """Current top-level 5-bool flags of an entry (after any human GT correction)."""
    return {d: bool(entry.get(d, False)) for d in DIRS}


def predicted_flags(entry, snapshot_key="predicted"):
    """The model's original prediction: prefer the first-correction snapshot
    (snapshot_key) if present, else the current top-level flags."""
    snap = entry.get(snapshot_key)
    if isinstance(snap, dict):
        return {d: bool(snap.get(d, False)) for d in DIRS}
    return read_flags(entry)


def vlm_of(entry):
    """Convenience for the VLM-era viewer: snapshot under 'vlm_predicted' else current."""
    return [d for d in DIRS if predicted_flags(entry, "vlm_predicted")[d]]


def validate_record(rec):
    """True if rec carries all 5 direction bools."""
    return all(d in rec for d in DIRS)
