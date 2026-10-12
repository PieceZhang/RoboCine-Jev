#!/usr/bin/env python3
"""scenario_gen.py — config-driven scenario runner (design B: pipeline-driven).

Reads a scenario YAML from cinema_traj/scenes/config/<scene>.yml containing
staging (cast/props) + semantic camera intent (beats) + audit bounds, and:

    1. writes plan.json            (semantic steps, no coordinates)
    2. runs the cinema_traj pipeline  build → optimize  (all camera
       coordinates are computed by the pipeline, never by this file)
    3. writes cast.json beside the trajectory
    4. audits the OPTIMISED trajectory against the OBB + person clearance
       model and hard-asserts min_clearance

The cast must also exist in the scene JSON as person_<id> obstacle OBBs
(otherwise the optimiser flies through people — it only sees the scene SDF);
the consistency of the two is checked on every run.

Usage:
    python tools/scenario_gen.py cinema_traj/scenes/config/kujiale_0021_sofa_trio.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vlm_intention.common.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# intent → plan.json
# --------------------------------------------------------------------------- #
# One-line-per-move semantic format (CameraMovementsRecommend-style fields).
# A move line looks like:
#   {运镜类型: 环绕, 屏幕方向: 横屏, 运镜方向: 右, 初始拍摄角度: 水平, 运镜节奏: 中}
# Everything not stated gets a documented default; optional override keys
# (distance, azimuth_deg, sweep_deg, swing_deg, close_ratio, duration_s)
# may be added to a line when a specific staging is wanted.
_MOVE_TEMPO_S = {"慢": 8.0, "缓": 8.0, "slow": 8.0,
                 "中": 6.0, "medium": 6.0,
                 "快": 4.0, "fast": 4.0}
_MOVE_ANGLE0 = {"水平": 0.0, "平视": 0.0, "level": 0.0,
                "仰视": -15.0, "仰": -15.0, "低": -15.0,
                "俯视": 20.0, "俯": 20.0, "高": 20.0}
_MOVE_TYPE = {  # semantic name → base type (routed in _move_to_beat).
    # truck words FIRST so the substring fallback prefers the longer,
    # more specific key (横移摇 → 横移 → truck, not 摇 → pan).
    "横移": "truck", "平移": "truck", "滑移": "truck", "横移跟拍": "truck",
    "truck": "truck", "track": "truck", "lateral": "truck",
    "横摇": "pan", "摇镜": "pan", "平摇": "pan", "摇": "pan", "pan": "pan",
    "环绕": "arc_swing", "环拍": "arc_swing", "弧摆": "arc_swing", "swing": "arc_swing",
    "后拉": "move_out", "拉远": "move_out", "后退": "move_out",
    "前推": "move_in", "推近": "move_in",
    "升降": "crane", "上升": "crane", "下降": "crane",
    "定格": "static", "静止": "static",
    "变焦": "zoom_in_out", "变焦推": "zoom_in_out",
    "变焦拉": "zoom_out_in",
}


def _move_to_beat(mv: dict) -> dict:
    """One semantic move line → {anchor, traj} beat with defaults."""
    raw = str(mv.get("运镜类型", mv.get("type", ""))).strip()
    base = _MOVE_TYPE.get(raw)
    if base is None:  # tolerant: substring match (环绕拍摄 → 环绕)
        for k, v in _MOVE_TYPE.items():
            if k in raw:
                base = v
                break
    if base is None:
        raise ValueError(f"moves 行无法识别运镜类型 '{raw}'"
                         f"（支持：{sorted(set(_MOVE_TYPE))}）")
    direction = str(mv.get("运镜方向", mv.get("direction", "右"))).strip()
    angle0 = _MOVE_ANGLE0.get(str(mv.get("初始拍摄角度", "水平")).strip(), 0.0)
    tempo = _MOVE_TEMPO_S.get(str(mv.get("运镜节奏", "中")).strip(), 6.0)
    duration = float(mv.get("duration_s", tempo))
    # screen orientation is consumed downstream by rendering (landscape
    # 1280x720 / portrait 720x1280); recorded in the doc, not the plan
    anchor = {"elevation": mv.get("elevation", "medium"),
              "distance": mv.get("distance", "far"),
              "start_angle_deg": angle0}
    traj = {"duration_s": duration}
    if base == "pan":
        # PAN (tripod semantics): camera position FIXED, the optical axis
        # sweeps horizontally — base parks, the arm/gimbal serves the yaw.
        # Round-trip sweep by default: a one-way pan leaves the gimbal
        # offset for the next transition's base park to unwind (measured
        # base-yaw teleport, cf. 0af57ae).
        traj["type"] = "pan_left" if direction in ("左", "left") else "pan_right"
        traj["sweep_deg"] = float(mv.get("sweep_deg", 45.0))
        if mv.get("height_s") is not None and mv.get("height_e") is not None:
            traj["height_s"] = float(mv["height_s"])
            traj["height_e"] = float(mv["height_e"])
        traj["sweep_return"] = bool(mv.get("sweep_return", True))
        traj["look_point"] = mv.get("look_point", "head")
        if mv.get("azimuth_deg") is not None:
            anchor["azimuth_deg"] = float(mv["azimuth_deg"])
    elif base == "truck":
        # LATERAL TRUCK (slider semantics): the camera TRANSLATES from
        # the left person to the right person along the sofa at corridor
        # distance, facing the subject (subject centered in frame) — the
        # spiral construction keeps look-vs-travel inside the arm window.
        traj["type"] = "lateral_truck"
        traj["direction"] = -1.0 if direction in ("左", "left") else 1.0
        traj["look_point"] = mv.get("look_point", "head")
        if mv.get("azimuth_deg") is not None:
            anchor["azimuth_deg"] = float(mv["azimuth_deg"])
    elif base == "arc_swing":
        # SUBJECT-CENTERED SPIRAL ARC: build.py's orbit branch swings the
        # planned sweep's chord around the subject with an analytically
        # solved radius close (worst look-vs-travel = 84°, 6° inside the
        # wrist branch flip) — dead-center framing, hard gates judge
        # per-waypoint reachability.
        traj["type"] = "orbit_quarter"
        traj["direction"] = -1.0 if direction in ("左", "left") else 1.0
        traj["sweep_deg"] = float(mv.get("sweep_deg",
                                         mv.get("swing_deg", 30.0)))
        anchor["azimuth_deg"] = float(mv.get("azimuth_deg", -15.0))
        traj["look_point"] = mv.get("look_point", "head")
    elif base in ("move_out", "move_in"):
        traj["type"] = base if base == "move_out" else "move_in"
        traj["look_point"] = mv.get("look_point", "head")
        # the pull-back anchors at its OWN frontal position for a REAL
        # pull (the continuity anchor shrank it to 0.35 m); subject
        # stays centered (look = subject head)
        anchor["azimuth_deg"] = float(mv.get("azimuth_deg", 0))
        if direction in ("前", "推", "forward") and base == "move_out":
            traj["type"] = "move_in"
        elif direction in ("后", "拉", "back") and base == "move_in":
            traj["type"] = "move_out"
    else:
        traj["type"] = base
        if mv.get("height_s") is not None and mv.get("height_e") is not None:
            traj["height_s"] = float(mv["height_s"])
            traj["height_e"] = float(mv["height_e"])
        # crane & friends: anchor azimuth routed like the other move
        # families (default FRONTAL — the proven south corridor for the
        # kujiale sofa; the raw anchor default parks southwest)
        anchor["azimuth_deg"] = float(mv.get("azimuth_deg", 0.0))
    return {"anchor": anchor, "traj": traj}


def scen_beats(scen: dict) -> list:
    """moves: (one semantic line per camera movement) take precedence;
    beats: (explicit anchor/traj dicts) stay supported for staging work."""
    moves = scen.get("moves")
    if moves:
        return [_move_to_beat(mv) for mv in moves]
    return scen.get("beats", [])


def beats_to_plan(scen: dict) -> dict:
    """Scenario beats ('anchor + optional transition + traj') expand to the
    1-3-1 plan-schema step sequence (plan_schema.validate_plan)."""
    target = scen["target"]
    steps = []
    for i, beat in enumerate(scen_beats(scen)):
        steps.append({"kind": "anchor", "target": target, **beat["anchor"]})
        if i > 0:  # transition belongs between this anchor and its traj
            trans = beat.get("transition", {"type": "arc", "alpha_deg": 0.0,
                                            "duration_s": 9.0})
            steps.append({"kind": "transition", "target": target, **trans})
        steps.append({"kind": "object_traj", "target": target, **beat["traj"]})
    for i, s in enumerate(steps):
        s["step"] = i + 1
    return {"plan_id": scen["plan_id"],
            "prompt": scen.get("prompt", ""),
            "model": "scenario_config",
            "user_specified_order": True,
            "object_sequence": [target],
            "steps": steps,
            "reasoning": f"scenario config: {scen['config_path'].name}"}


# --------------------------------------------------------------------------- #
# cast ↔ scene consistency (persons live in the scene as person_<id> obstacle
# OBBs so the optimiser can never fly through them; the yml seats must match)
# --------------------------------------------------------------------------- #
PERSON_BOX = (0.7, 0.44)  # planning OBB full extents: width × depth (m)


def check_cast_obstacles(scen: dict, scene: dict) -> None:
    for c in scen.get("cast", []):
        want = [c["seat"][0], c["seat"][1] + float(c["height_m"]) / 2,
                c["seat"][2]]
        obj = next((o for o in scene.get("objects", [])
                    if o.get("id") == f"person_{c['id']}"), None)
        if obj is None:
            print(f"  ⚠️  scene lacks obstacle person_{c['id']} — optimiser "
                  "will not avoid this figure; add it to the scene JSON")
            continue
        got = obj["obb"]["center"]
        drift = float(np.linalg.norm(np.array(want, float) - np.array(got)))
        if drift > 0.01:
            print(f"  ⚠️  person_{c['id']}: yml seat vs scene OBB centre "
                  f"drift {drift:.3f} m — keep the two in sync")


def write_casts(scen: dict, out_dir: Path) -> None:
    if not scen.get("cast") and not scen.get("props"):
        return                       # subject-only scenario: no cast.json
    chars = [{k: v for k, v in c.items() if k != "height_m"}
             for c in scen.get("cast", [])]
    doc = {"_comment": scen.get("cast_comment", ""), "characters": chars}
    if scen.get("props"):
        doc["props"] = scen["props"]
    (out_dir / "cast.json").write_text(json.dumps(doc, indent=1,
                                                  ensure_ascii=False),
                                       encoding="utf-8")
    print(f"  cast.json  ({len(chars)} figures, "
          f"{len(scen.get('props', []))} props)")


# --------------------------------------------------------------------------- #
# audit of the optimised trajectory
# --------------------------------------------------------------------------- #
def audit_clearance(frames: list[dict], scen: dict) -> dict:
    a = scen["audit"]
    obbs = [(o["name"], np.array(o["center"]), np.array(o["half"]))
            for o in a.get("obbs", [])]
    persons = [(c["id"], np.array(c["seat"]), float(c["height_m"]))
               for c in scen.get("cast", [])]
    band_lo, band_hi = a["arm_band"]

    def box_clear(pt, c, e):
        d = np.abs(pt - c) - e
        if (d < 0).all():
            return float(d.max())
        return float(np.linalg.norm(np.maximum(d, 0.0)))

    worst: dict[str, tuple[float, str]] = {}
    for f in frames:
        pt = np.array(f["pos"], float)
        cands = [(box_clear(pt, c, e), n) for n, c, e in obbs]
        for nm, p, h in persons:
            ex = a["arm_half"] if p[1] + band_lo <= pt[1] <= p[1] + min(band_hi, h) \
                else a["torso_half"]
            cands.append((box_clear(pt, p + [0, h / 2, 0],
                                    [ex, h / 2, a["depth_half"]]),
                          f"person:{nm}"))
        cl, hit = min(cands, key=lambda t: t[0])
        seg = f["segment"]
        if seg not in worst or cl < worst[seg][0]:
            worst[seg] = (round(cl, 3), hit)
    for seg, (cl, hit) in worst.items():
        print(f"  audit {seg:24s} min clearance {cl:+.3f} m  ({hit})")
    bad = {k: v for k, v in worst.items() if v[0] < a["min_clearance"]}
    assert not bad, (f"scenario audit FAILED (< {a['min_clearance']} m): {bad} "
                     "— adjust the intent beats/anchors, not numbers")
    return worst


# --------------------------------------------------------------------------- #
def check_assets_exist(scen: dict) -> None:
    """Cast/props binaries are NOT committed to the repo (NVIDIA Isaac Sim
    asset licence — they are generated locally from the user's pack). Fail
    early with an actionable message instead of a FileNotFoundError deep in
    the simview cast loader."""
    paths = [c.get("usda") for c in scen.get("cast", [])]
    paths += [p.get("usd") for p in scen.get("props", [])]
    missing = [p for p in paths if p and not (REPO / p).exists()]
    assert not missing, (
        "scenario assets missing:\n  - " + "\n  - ".join(missing) +
        "\n→ these are generated locally from your Isaac_People pack into "
        "datasets/simview_assets/ (they are not in git; see README scenario "
        "demo section)")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scenario", help="cinema_traj/scenes/config/<scene>.yml")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="only rewrite plan/cast files + audit existing "
                         "traj_final.json (no build/optimize rerun)")
    args = ap.parse_args()

    cfg_path = Path(args.scenario).resolve()
    scen = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    scen["config_path"] = cfg_path
    scene_id = scen["scene_id"]
    scene_path = REPO / "cinema_traj" / "scenes" / f"{scene_id}.json"
    scene_doc = json.loads(scene_path.read_text(encoding="utf-8"))
    staging = scene_doc.get("staging", {})

    # MINIMAL-INPUT mode: a yml with `moves:` carries ONLY the camera
    # intent (scene_id + subject + one line per move). Staging (cast/props/
    # audit) then comes from the scene json's "staging" field; the target
    # is the scene's subject_track; plan_id defaults to the scene id.
    if scen.get("moves") and "cast" not in scen:
        scen.setdefault("cast", staging.get("cast", []))
        scen.setdefault("props", staging.get("props", []))
        if "audit" in staging:
            scen.setdefault("audit", staging["audit"])
    if "target" not in scen:
        st = scene_doc.get("subject_track") or {}
        assert st.get("subject_id"), (
            f"{scene_path.name}: yml 无 target/subject，场景也无 subject_track")
        scen["target"] = st["subject_id"]
    scen.setdefault("plan_id", scene_id)

    check_assets_exist(scen)
    out_dir = REPO / "cinema_traj_out" / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()  # default outputs_dir = cinema_traj_out

    # 1. intent → plan.json (no coordinates)
    plan = beats_to_plan(scen)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1,
                                                  ensure_ascii=False),
                                       encoding="utf-8")
    print(f"  plan.json  ({len(plan['steps'])} steps, "
          f"{len(scen_beats(scen))} moves/beats, semantic only)")

    # 2. pipeline computes ALL camera coordinates — on the scene itself,
    #    which carries the cast as person_* obstacle OBBs
    scene = scene_doc
    if scen.get("cast"):
        check_cast_obstacles(scen, scene)
    if not args.skip_pipeline:
        from cinema_traj.stages import build, optimize
        opt = scen.get("optimizer", {})
        if opt:  # scenario's optimizer profile → cinema_traj.optimizer (OptCfg)
            cfg.setdefault("cinema_traj", {}).setdefault(
                "optimizer", {}).update(opt)
        build.run(cfg, scene_path=str(scene_path),
                  plan_path=str(out_dir / "plan.json"), force=True)
        optimize.run(cfg, scene_path=str(scene_path), force=True)

    # 3. staging
    write_casts(scen, out_dir)

    # 4. audit the pipeline output
    traj = json.loads((out_dir / "traj_final.json").read_text(encoding="utf-8"))
    m = traj.get("metrics", {})
    print(f"  traj_final.json  {m.get('frames')} frames  "
          f"collision {m.get('collision_rate')}  "
          f"min scene clearance {m.get('min_clearance_m')} m")
    if "audit" in scen:
        audit_clearance(traj["frames"], scen)
    else:
        print("  (no audit block — scenario without staged obstacles)")
    print(f"✅ scenario '{scene_id}' ready → {out_dir}")


if __name__ == "__main__":
    main()
