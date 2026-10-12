"""Trajectory plan schema + 1-3-1 validator (paper §3.2.2, appendix E).

A plan for N visitation targets (objects and/or the subject) is exactly
3N−1 steps:

    A₁ → T_o,1 → (A₂ → T_t → T_o,₂)* → … → A_N → T_t,(N−1)→N → T_o,N

Each step is a plain dict (JSON-serializable):

    {"step": 1, "kind": "anchor"|"object_traj"|"subject_traj"|"transition",
     "target": "sofa_01" | "subject",
     "type": "orbit_half" | ... (object_traj/subject_traj/transition only),
     "elevation": "medium", "distance": "close",     # anchors
     "duration_s": 6.0, "alpha_deg": 0.0}            # trajectories / arcs
"""

from __future__ import annotations

from ..common import constants as C
from ..common.scene import Scene

STEP_KINDS = ("anchor", "object_traj", "subject_traj", "transition")
ELEVATIONS = tuple(C.ELEVATION_BANDS_DEG.keys())
DISTANCES = tuple(C.DISTANCE_MULTIPLIERS.keys())

# zh ← en aliases emitted by the LLM
_ALIASES = {
    "运镜": "kind", "类型": "kind", "目标": "target", "物体": "target",
    "高度": "elevation", "视角": "elevation", "距离": "distance",
    "时长": "duration_s", "弧度": "alpha_deg",
    "锚点": "anchor", "锚定": "anchor", "anchor": "anchor",
    "object_traj": "object_traj", "主体运镜": "subject_traj",
    "过渡": "transition", "过渡运镜": "transition", "arc": "transition",
    "近": "close", "中": "medium", "远": "far",
    "低": "low", "中高": "high", "高": "high", "俯视": "overhead", "顶视": "overhead",
}


def normalize_steps(raw: list) -> list:
    """Key aliases → canonical schema; drop unknown fields; number steps."""
    steps = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            continue
        d = {}
        for k, v in s.items():
            kk = _ALIASES.get(str(k).strip().lower(), str(k).strip())
            if isinstance(v, str):
                v = _ALIASES.get(v.strip(), v.strip())
            d[kk] = v
        d.setdefault("step", i + 1)
        d["step"] = i + 1
        steps.append(d)
    return steps


def normalize_plan(raw: dict, scene: Scene) -> dict:
    plan = {
        "plan_id": raw.get("plan_id", "plan"),
        "prompt": raw.get("prompt", ""),
        "model": raw.get("model", "unknown"),
        "user_specified_order": bool(raw.get("user_specified_order", True)),
        "object_sequence": list(raw.get("object_sequence", raw.get("objects", []))),
        "steps": normalize_steps(raw.get("steps", raw.get("trajectory", []))),
        "reasoning": raw.get("reasoning", ""),
    }
    # canonicalize the subject alias to the scene's actual subject_id
    subject_id = scene.subject.subject_id if scene.subject is not None else "subject"
    for s in plan["steps"]:
        if s.get("target") in ("subject", "主体", "演员"):
            s["target"] = subject_id
    plan["object_sequence"] = [subject_id if t in ("subject", "主体", "演员") else t
                               for t in plan["object_sequence"]]
    if not plan["object_sequence"]:
        seq = []
        for s in plan["steps"]:
            if s.get("kind") == "anchor" and s.get("target") not in seq:
                seq.append(s.get("target"))
        plan["object_sequence"] = seq
    return plan


def validate_plan(plan: dict, scene: Scene) -> list[str]:
    """Structural + scene-grounding checks; returns human-readable errors
    (fed back to the LLM for one repair round)."""
    errs: list[str] = []
    steps = plan.get("steps", [])
    subject_id = scene.subject.subject_id if scene.subject is not None else "subject"
    known = {o.id for o in scene.objects} | ({subject_id} if scene.subject else set())

    # --- targets grounded -------------------------------------------------- #
    for i, s in enumerate(steps):
        tgt = s.get("target")
        if s.get("kind") in ("anchor", "object_traj", "subject_traj") and tgt not in known:
            errs.append(f"step {i + 1}: unknown target '{tgt}' (known: {sorted(known)})")

    # --- 1-3-1 pattern ------------------------------------------------------ #
    anchors = [s for s in steps if s.get("kind") == "anchor"]
    obj_trajs = [s for s in steps if s.get("kind") in ("object_traj", "subject_traj")]
    trans = [s for s in steps if s.get("kind") == "transition"]
    if anchors or obj_trajs:
        n = len(anchors)
        if n == 0:
            errs.append("plan has no anchor steps")
        elif n != len(obj_trajs):
            errs.append(f"expected {n} trajectory steps to match {n} anchors, "
                        f"got {len(obj_trajs)}")
        elif len(trans) != n - 1:
            errs.append(f"expected {n - 1} transition steps, got {len(trans)}")
        else:
            walked = _walk_pattern(steps)
            if walked != n:
                errs.append(f"steps do not follow the 1-3-1 alternation "
                            f"(anchor → traj → [anchor → transition → traj]*); "
                            f"{walked}/{n} groups well-formed")

    # --- per-step fields ---------------------------------------------------- #
    for i, s in enumerate(steps):
        kind = s.get("kind")
        if kind not in STEP_KINDS:
            errs.append(f"step {i + 1}: kind must be one of {STEP_KINDS}")
            continue
        if kind == "anchor":
            if s.get("elevation") and s["elevation"] not in ELEVATIONS:
                errs.append(f"step {i + 1}: elevation must be one of {ELEVATIONS}")
            if s.get("distance") and s["distance"] not in DISTANCES:
                errs.append(f"step {i + 1}: distance must be one of {DISTANCES}")
        else:
            t = s.get("type")
            allowed = (C.OBJECT_TRAJ_TYPES if kind == "object_traj" else
                       C.SUBJECT_TRAJ_TYPES if kind == "subject_traj" else
                       C.TRANSITION_TRAJ_TYPES)
            if t not in allowed:
                errs.append(f"step {i + 1}: type '{t}' invalid for kind '{kind}' "
                            f"(allowed: {allowed})")
            dur = s.get("duration_s", C.DEFAULT_DURATION)
            if not isinstance(dur, (int, float)) or dur <= 0 or dur > 60:
                errs.append(f"step {i + 1}: duration_s must be in (0, 60] seconds")

    # --- placement constraints (paper appendix E.2) -------------------------- #
    for i, s in enumerate(steps):
        if s.get("kind") != "object_traj":
            continue
        obj = scene.object_by_id(s.get("target", ""))
        if obj is None:
            continue
        forbidden = C.PLACEMENT_FORBIDDEN.get(obj.placement, set())
        if s.get("type") in forbidden:
            errs.append(f"step {i + 1}: target '{obj.id}' has placement "
                        f"'{obj.placement}' — movement '{s['type']}' is forbidden")

    # --- subject grounding --------------------------------------------------- #
    if scene.subject is None:
        for i, s in enumerate(steps):
            if s.get("kind") == "subject_traj":
                errs.append(f"step {i + 1}: subject step but the scene has no subject_track")
    return errs


def _walk_pattern(steps: list) -> int:
    """Count well-formed (anchor, [transition,] traj) groups in order."""
    i, groups = 0, 0
    kinds = [s.get("kind") for s in steps]
    while i < len(kinds):
        if kinds[i] != "anchor":
            break
        i += 1
        if i < len(kinds) and kinds[i] == "transition":
            i += 1
        if i >= len(kinds) or kinds[i] not in ("object_traj", "subject_traj"):
            break
        i += 1
        groups += 1
    ok = i == len(kinds)
    return groups if ok else 0
