#!/usr/bin/env python3
"""Replica semantic transfer: frl_apartment_0 mesh instance labels → our 3DGS
cloud → refit OBBs → autosem scene JSON (a COPY; the hand-refined scene file
is never modified).

Frame facts (verified P0: median NN residual 2.25 cm, zero systematic offset):
  mesh_semantic.ply frame  ==  raw PLY disk frame (mcmc_3dgs, Z-up flavor)
  internal scene frame     =   _to_yup(raw, "z") then y += 1.624 (floor = 0)

Usage:
    python scripts/replica_autosem.py            # south living room (mentor's scope)
    python scripts/replica_autosem.py --full     # whole apartment footprint
Outputs:
    cinema_traj/scenes/frl_apartment_0_autosem[_full].json   (+ transfer stats)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cinema_traj.common.obb import OBB, fit_obb_pca
from cinema_traj.stages.extract import load_points, _to_yup, voxel_downsample, \
    classify_placement

MESH = REPO / "datasets/replica_v1/frl_apartment_0/habitat/mesh_semantic.ply"
INFO = REPO / "datasets/replica_v1/frl_apartment_0/habitat/info_semantic.json"
CLOUD = REPO / "datasets/replica_mcmc_3dgs/frl_apartment_0/ply/point_cloud_30000.ply"
REFINED = REPO / "cinema_traj/scenes/frl_apartment_0.json"
AUTOSEM = REPO / "cinema_traj/scenes/frl_apartment_0_autosem.json"

FULL = "--full" in sys.argv
SUFFIX = "_full" if FULL else ""
OUT = REPO / f"cinema_traj/scenes/frl_apartment_0_autosem{SUFFIX}.json"

# classes that become room STRUCTURE in --full mode (interior walls matter!)
PARTITION_CLASSES = {"wall", "door"}            # door closed ≈ wall for collision

VSHIFT = 1.624            # refined-scene floor=0 shift (provenance.map_shift)
NN_GATE_M = 0.05          # label transfer radius
SMOOTH_VOX = 0.07         # majority-vote voxel
MIN_PTS = 80              # keep instances with at least this many cloud points
MATCH_DIST = 0.9          # center distance to adopt a hand-refined label/id
STRUCTURE_CLASSES = {"wall", "floor", "ceiling", "door", "blinds", "window",
                     "handrail", "pillar", "undefined", ""}

# flat decorative classes — NEVER collision obstacles. Replica anonymizes
# personal photos as anonymize_picture; their instances transfer with
# imprecise positions and end up as floating panels mid-room at head height,
# which poisons orbit feasibility (PR #2 known issue). The cloud still
# renders them; they just don't enter the scene graph.
DECOR_CLASSES = {"anonymize_picture", "picture", "poster", "painting"}


# --------------------------------------------------------------------------- #
# mesh_semantic.ply parsing (binary LE; verts 27B stride; faces u8 n / n*u32 / u16 id)
# --------------------------------------------------------------------------- #
def load_semantic_mesh(path: Path):
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fh.read(1)
        h = header.decode()
        nV = int([l for l in h.splitlines() if l.startswith("element vertex")][0].split()[-1])
        buf = fh.read()
    vdt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    verts = np.frombuffer(buf[:nV * 27], dtype=vdt)
    xyz = np.stack([verts['x'], verts['y'], verts['z']], 1)

    # per-vertex instance id = majority over incident faces
    from collections import Counter, defaultdict
    acc = defaultdict(Counter)
    i = nV * 27
    end = len(buf)
    while i < end:
        n = buf[i]; i += 1
        vids = np.frombuffer(buf, dtype='<u4', count=n, offset=i); i += 4 * n
        oid = int(np.frombuffer(buf, dtype='<u2', count=1, offset=i)[0]); i += 2
        for v in vids:
            acc[int(v)][oid] += 1
    vert_oid = np.full(nV, -1, dtype=np.int64)
    for v, c in acc.items():
        vert_oid[v] = c.most_common(1)[0][0]
    return xyz, vert_oid


def main() -> None:
    info = json.load(open(INFO))
    id2name = {o["id"]: o["class_name"] for o in info["objects"]}

    mesh_xyz, vert_oid = load_semantic_mesh(MESH)
    ok = vert_oid >= 0
    print(f"mesh: {len(mesh_xyz)} verts, {int(ok.sum())} labeled, "
          f"{len(set(vert_oid[ok].tolist()))} instances")

    # ---- cloud into internal frame (Y-up, floor=0) ----
    raw = load_points(str(CLOUD), up="y")
    cloud = _to_yup(raw, up="z")
    cloud[:, 1] += VSHIFT
    sub = voxel_downsample(cloud, 0.05)
    # transfer labels in the MESH frame: same geometric transform inverse
    sub_mesh = _to_yup(sub, up="y")           # _to_yup is an involution pair? verify:
    sub_mesh = np.stack([sub[:, 0], -sub[:, 2], sub[:, 1] - VSHIFT], 1)  # exact inverse of (x,z,-y)+shift
    tree = cKDTree(mesh_xyz[ok])
    dist, ix = tree.query(sub_mesh)
    inst = np.where(dist < NN_GATE_M, vert_oid[ok][ix], -1)
    print(f"transfer coverage @{NN_GATE_M*100:.0f}cm: {(inst>=0).mean()*100:.1f}%")

    # ---- majority smoothing in voxels ----
    keys = np.floor(sub / SMOOTH_VOX).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    smooth = np.full(len(sub), -1, dtype=np.int64)
    for vi in range(len(uniq)):
        ids = inst[inv == vi]
        ids = ids[ids >= 0]
        if len(ids):
            smooth[inv == vi] = np.bincount(ids).argmax()
    print(f"smoothed coverage: {(smooth>=0).mean()*100:.1f}%")

    # ---- region + structure source: refined (scoped) vs FULL (whole cloud) ----
    scene = json.load(open(REFINED))
    if FULL:
        lo, hi = sub.min(axis=0), sub.max(axis=0)
        rmin, rmax = lo - 0.1, hi + 0.1                    # whole apartment
        inroom = np.all((sub >= rmin) & (sub <= rmax), axis=1)
        # replicate the refined shells at full footprint: floor/ceiling heights
        # proven by the refined scene (floor 0 / ceiling ≈2.85 m)
        full_floor = full_ceil = None
        for s in scene["structures"]:
            c, e = np.array(s["obb"]["center"]), np.array(s["obb"]["extents"])
            if s["kind"] == "floor":
                full_floor = c[1] + e[1] / 2
            if s["kind"] == "ceiling":
                full_ceil = c[1] + e[1] / 2
        assert full_floor is not None and full_ceil is not None, \
            "refined scene must carry floor/ceiling structures for --full"
        mid = (rmin + rmax) / 2
        xc, zc = float(mid[0]), float(mid[2])
        structures = [
            {"id": "floor", "kind": "floor", "room": "room_01",
             "obb": OBB(np.array([xc, full_floor - 0.025, zc]), np.eye(3),
                        np.array([rmax[0] - rmin[0], 0.05, rmax[2] - rmin[2]])).to_dict()},
            {"id": "ceiling", "kind": "ceiling", "room": "room_01",
             "obb": OBB(np.array([xc, full_ceil + 0.025, zc]), np.eye(3),
                        np.array([rmax[0] - rmin[0], 0.05, rmax[2] - rmin[2]])).to_dict()}]
        ceil_y = full_ceil
        rooms = [{"id": "room_01", "label": "full_apartment",
                 "bounds_min": rmin.tolist(), "bounds_max": rmax.tolist()}]
        walls = []
    else:
        rmin = np.array(scene["rooms"][0]["bounds_min"]) - 0.3
        rmax = np.array(scene["rooms"][0]["bounds_max"]) + 0.3
        inroom = np.all((sub >= rmin) & (sub <= rmax), axis=1)
        walls = []
        ceil_y = None
        for s in scene["structures"]:
            c, e = np.array(s["obb"]["center"]), np.array(s["obb"]["extents"])
            if s["kind"] == "ceiling":
                ceil_y = c[1] + e[1] / 2
            if s["kind"] == "wall":
                walls.append((int(np.argmin(e)), float(c[int(np.argmin(e))]), None))

    # ---- per-instance refit ----
    objs = []
    for oid in sorted(set(smooth[inroom & (smooth >= 0)].tolist())):
        cls = id2name.get(int(oid), "object")
        if cls in STRUCTURE_CLASSES and not (FULL and cls in PARTITION_CLASSES):
            continue
        if cls in DECOR_CLASSES:
            continue        # flat decor — visual-only, see DECOR_CLASSES note
        if FULL and cls == "object":
            continue        # catch-all pseudo-instances span the whole apartment
                            # and would poison the collision field
                            # ("undefined" is already in STRUCTURE_CLASSES)
        pts = sub[inroom & (smooth == oid)]
        if len(pts) < MIN_PTS:
            continue
        obb = fit_obb_pca(pts)
        objs.append({"oid": int(oid), "cls": cls, "obb": obb, "n": int(len(pts))})

    if FULL:                                                # partitions → structures
        parts = [o for o in objs if o["cls"] in PARTITION_CLASSES]
        objs = [o for o in objs if o["cls"] not in PARTITION_CLASSES]
        for p in parts:
            structures.append({"id": f"wall_{p['oid']}", "kind": "wall",
                               "room": "room_01", "obb": p["obb"].to_dict()})
            e = np.array(p["obb"].extents)
            # argmin(extents) is the LOCAL PCA axis; classify_placement wants the
            # WORLD axis of the wall normal — map through the OBB rotation
            thin = int(np.argmin(e))
            waxis = int(np.argmax(np.abs(np.array(p["obb"].rotation)[:, thin])))
            walls.append((waxis,
                          float(np.array(p["obb"].center)[waxis]), None))
        print(f"partition structures added: {len(parts)}")
    print(f"auto instances kept: {len(objs)}")

    # ---- autosem objects keep their Replica identity (no label takeover);
    #      hand-box correspondence is emitted as analysis only (_near_hand) ----
    hand = scene["objects"]
    out_objects = []
    for j, o in enumerate(objs):
        placement = classify_placement(o["obb"], walls, ceil_y)
        slug = o["cls"].replace("-", "_")
        entry = {"id": f"{slug}_{o['oid']:03d}", "label": o["cls"],
                 "room": "room_01", "obb": o["obb"].to_dict(),
                 "placement": placement, "_src": f"replica:{o['cls']}#{o['oid']}"}
        # nearest hand-refined box (analysis annotation, never a label override)
        bd, bh = 1e9, None
        for ho in hand:
            d = float(np.linalg.norm(o["obb"].center - np.array(ho["obb"]["center"])))
            if d < bd:
                bd, bh = d, ho["id"]
        if bh is not None and bd < 2.0:
            entry["_near_hand"] = {"id": bh, "drift_m": round(bd, 3)}
        out_objects.append(entry)
    n_extra = len(out_objects)

    # ---- assemble autosem scene ----
    auto = dict(scene)
    auto["scene_id"] = f"frl_apartment_0_autosem{SUFFIX}"
    auto["objects"] = out_objects
    if FULL:
        auto["rooms"] = rooms
        auto["structures"] = structures
        # subject: prefer the current autosem track (the verified N-S route),
        # else fall back to the hand scene's
        auto["subject_track"] = json.load(open(AUTOSEM)).get("subject_track") \
            if AUTOSEM.exists() else scene.get("subject_track")
    auto["provenance"] = {**scene.get("provenance", {}),
                          "autosem": {
                              "source": "Replica v1 mesh_semantic.ply NN transfer",
                              "nn_gate_m": NN_GATE_M, "smooth_vox_m": SMOOTH_VOX,
                              "min_pts": MIN_PTS, "match_dist_m": MATCH_DIST,
                              "auto_objects": n_extra, "full_room": FULL,
                              "note": ("FULL: whole-apartment footprint, floor/ceiling "
                                       "re-extended, Replica wall/door instances as "
                                       "structures; " if FULL else
                                       "rooms/structures/subject copied from the "
                                       "hand-refined scene; ") +
                                      "objects refit from Replica labels"}}
    json.dump(auto, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"scene: {OUT}\n  objects: {len(out_objects)} (Replica-labeled; "
          f"hand correspondence annotated in _near_hand, analysis only)")
    for o in out_objects:
        c = np.round(np.array(o["obb"]["center"]), 2)
        e = np.round(np.array(o["obb"]["extents"]), 2)
        nh = o.get("_near_hand")
        print(f"  {o['id']:<20} {o['label'][:22]:<24} {o['_src']:<26} "
              f"c={c} e={e} {o['placement']:<12}"
              + (f"  ≈hand:{nh['id']} drift={nh['drift_m']}m" if nh else ""))


if __name__ == "__main__":
    main()
