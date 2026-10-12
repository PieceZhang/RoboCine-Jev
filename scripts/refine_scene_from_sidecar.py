#!/usr/bin/env python3
"""Refine an extract draft scene using the usd_to_ply.py semantic sidecar.

The extract stage is geometry-only: generic object_NN labels, double-detected
walls (front/back faces peak separately), one big room. InteriorAgent gives us
exact per-asset OBBs + semantic names (bed / sofa / door …), so this script
rebuilds the draft into the canonical hand-refined equivalent. Output defaults
to ``<stem>_traj.json`` — the plain ``<stem>.json`` slot may already be owned
by another pipeline (e.g. the simview dense scene for kujiale_0021):

  rooms       ← sidecar rooms.json polygons (converted to the internal frame)
  structures  ← sidecar wall/floor/ceiling assets (exact USD geometry)
  doors       ← sidecar door assets (kept OUT of objects so they don't block
                passage routing as collision obstacles)
  objects     ← sidecar object assets above a size threshold, semantic labels
                (repeated labels get _NN suffixes), placement classified
                against the sidecar walls

Usage:
    python scripts/refine_scene_from_sidecar.py \
        --scene cinema_traj/scenes/kujiale_0021_draft.json \
        --sidecar datasets/interioragent/kujiale_0021/kujiale_0021_sidecar.json \
        [--out cinema_traj/scenes/kujiale_0021_traj.json] \
        [--min-extent 0.35] [--max-objects 45] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vlm_intention.common.atomic_io import atomic_write_json
from cinema_traj.common.obb import OBB
from cinema_traj.common.scene import PLACEMENTS, STRUCTURE_KINDS
from cinema_traj.stages.extract import classify_placement

DOOR_LABELS = {"door", "doorsill"}
SCOPE_ROOM_NORMALIZE = {"livingroom": "living_room"}


def _obb(d: dict) -> OBB:
    return OBB(d["center"], d["rotation"], d["extents"])


def _room_of(center: np.ndarray, rooms: list, fallback: str = "") -> str:
    for r in rooms:
        lo, hi = r["bounds_min"], r["bounds_max"]
        if (lo[0] <= center[0] <= hi[0]) and (lo[2] <= center[2] <= hi[2]):
            return r["label"]
    if fallback:
        return fallback
    # outside every room polygon (exterior wall slabs etc.) → nearest room
    return min(rooms, key=lambda r: np.linalg.norm(
        (np.asarray(r["bounds_min"]) + np.asarray(r["bounds_max"])) / 2 - center))["label"]


def _walls_for_placement(structures: list) -> list:
    """[(axis, coord, None)] tuples for extract.classify_placement()."""
    walls = []
    for s in structures:
        if s["kind"] != "wall":
            continue
        obb = _obb(s["obb"])
        axis = int(np.argmin(obb.extents))
        if axis == 1:                       # horizontal slab — not a wall
            continue
        walls.append((axis, float(obb.center[axis]), None))
    return walls


def main() -> None:
    ap = argparse.ArgumentParser(description="语义边车 → 精修场景 JSON")
    ap.add_argument("--scene", required=True, help="extract draft scene JSON")
    ap.add_argument("--sidecar", required=True, help="usd_to_ply.py 的边车 JSON")
    ap.add_argument("--out", help="输出（默认 <draft stem 去掉 _draft>.json）")
    ap.add_argument("--min-extent", type=float, default=0.35,
                    help="物件 OBB 最大边下限（m），过滤杯子/勺子级碎屑")
    ap.add_argument("--min-samples", type=float, default=60.0,
                    help="边车采样点数下限")
    ap.add_argument("--max-objects", type=int, default=45)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    draft = json.loads(Path(a.scene).read_text())
    side = json.loads(Path(a.sidecar).read_text())
    # scene references (structure.room / object.room / doors.rooms) use the
    # room ID — make it the semantic label so plans read 'living_room', not 'room_01'
    rooms = [{**r, "id": r["label"]} for r in side["rooms"]]
    out_path = Path(a.out) if a.out else Path(a.scene).with_name(
        Path(a.scene).stem.replace("_draft", "") + "_traj.json")
    if out_path.exists() and not a.force:
        raise FileExistsError(f"{out_path} exists (use --force to overwrite)")

    structures, doors, raw_objects = [], [], []
    for asset in side["assets"]:
        obb = _obb(asset["obb_internal"])
        label = asset["label"]
        scope = asset["scope"]
        if asset["kind"] == "structure":
            room = _room_of(obb.center, rooms)
            structures.append({"kind": scope if scope in STRUCTURE_KINDS else "wall",
                               "room": room, "obb": asset["obb_internal"]})
        elif label in DOOR_LABELS:
            if label == "door":
                doors.append({"obb": asset["obb_internal"]})
        else:
            raw_objects.append((label, scope, asset, obb))

    # structures: group by kind, stable ids
    struct_entries, counts = [], {}
    for s in sorted(structures, key=lambda s: (s["kind"], -_obb(s["obb"]).extents.max())):
        counts[s["kind"]] = counts.get(s["kind"], 0) + 1
        sid = s["kind"] if counts[s["kind"]] == 1 else f"{s['kind']}_{counts[s['kind']]:02d}"
        struct_entries.append({"id": sid, "kind": s["kind"], "room": s["room"],
                               "obb": s["obb"]})
    walls = _walls_for_placement(struct_entries)
    ceiling_y = next((_obb(s["obb"]).center[1] for s in struct_entries
                      if s["kind"] == "ceiling"), None)

    # objects: size filter, label dedupe, placement + room
    kept = [o for o in raw_objects
            if o[3].extents.max() >= a.min_extent and o[2]["n_samples"] >= a.min_samples]
    kept.sort(key=lambda o: -o[3].extents.max())
    if len(kept) > a.max_objects:
        print(f"   ⚠️  {len(kept)} objects above threshold → keep largest {a.max_objects}")
        kept = kept[:a.max_objects]
    seen, objects = {}, []
    for label, scope, asset, obb in kept:
        seen[label] = seen.get(label, 0) + 1
        oid = label if seen[label] == 1 else f"{label}_{seen[label]:02d}"
        placement = classify_placement(obb, walls, ceiling_y)
        scope_room = SCOPE_ROOM_NORMALIZE.get(scope.rsplit("_", 1)[0],
                                              scope.rsplit("_", 1)[0]) \
            if scope != "other" else ""
        hint = SCOPE_ROOM_NORMALIZE.get(asset["room_hint"], asset["room_hint"]) \
            or scope_room
        room = _room_of(obb.center, rooms, fallback=hint or "living_room")
        objects.append({"id": oid, "label": oid, "room": room,
                        "obb": asset["obb_internal"], "placement": placement
                        if placement in PLACEMENTS else "freestanding"})

    # doors: nearest two rooms by center distance
    door_entries = []
    for i, d in enumerate(sorted(doors, key=lambda d: -_obb(d["obb"]).extents.max()), 1):
        c = _obb(d["obb"]).center
        dists = sorted(rooms, key=lambda r: np.linalg.norm(
            (np.asarray(r["bounds_min"]) + np.asarray(r["bounds_max"])) / 2 - c))
        door_entries.append({"id": f"door_{i:02d}", "kind": "door",
                             "rooms": [dists[0]["label"], dists[1]["label"]],
                             "obb": d["obb"]})

    scene = {
        "scene_id": out_path.stem,
        "rooms": rooms,
        "structures": struct_entries,
        "objects": objects,
        "doors": door_entries,
        "provenance": {
            **draft.get("provenance", {}),
            "extract_draft": str(a.scene),
            "refinement": [
                "objects/structures/doors rebuilt from the InteriorAgent USD "
                "sidecar (scripts/usd_to_ply.py): semantic labels + exact "
                "per-asset OBBs; draft walls (double front/back-face peaks) "
                "replaced by the real wall assets",
                f"object filter: max-extent ≥ {a.min_extent} m, "
                f"n_samples ≥ {a.min_samples:.0f}, top {len(objects)} by extent",
            ],
        },
    }
    atomic_write_json(out_path, scene)
    print(f"✅ refined scene → {out_path}")
    print(f"   rooms={len(rooms)} structures={len(struct_entries)} "
          f"objects={len(objects)} doors={len(door_entries)}")
    for o in objects[:12]:
        print(f"   · {o['id']:<20} room={o['room']:<12} placement={o['placement']}")


if __name__ == "__main__":
    main()
