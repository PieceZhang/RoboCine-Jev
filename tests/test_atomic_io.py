"""Atomic IO + schema round-trip."""
import json

from vlm_intention.common.atomic_io import atomic_write_json, load_json, backup_once
from vlm_intention.common.schema import DIRS, build_annotation_record, read_flags, predicted_flags


def test_atomic_roundtrip(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1})
    assert load_json(p) == {"a": 1}
    assert load_json(tmp_path / "missing.json", default={}) == {}


def test_backup_once_idempotent(tmp_path):
    p = tmp_path / "x.json"; p.write_text("{}")
    b1 = backup_once(p); b2 = backup_once(p)
    assert b1 and b2 is None          # only first call backs up


def test_record_roundtrip_and_snapshot():
    rec = build_annotation_record({"前": True, "下": True}, "x.png", "p/x.png", "m", source="t")
    assert rec["前"] and rec["下"] and not rec["左"]
    assert rec["image_name"] == "x.png" and rec["source"] == "t"
    assert set(read_flags(rec)) == set(DIRS)
    # after a simulated correction, predicted_flags falls back to current flags
    rec["predicted"] = {d: False for d in DIRS}
    assert predicted_flags(rec)["前"] is False      # snapshot preferred
