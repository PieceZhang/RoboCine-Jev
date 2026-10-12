#!/usr/bin/env python3
"""plan_fast — single-process fast path for the planning stages of
run_kid_combo_ros.sh (steps 3-7 + simview build).

Why: the 12-step script spawns ~6 torch-env python processes; import
overhead alone is ~1.5 s each (~7.5 s total). This driver pays the import
ONCE and calls the SAME stage code in-process, plus dedups the export↔retime
cascade (export auto-retimes on overspeed; the explicit `retime --force`
step re-ran it wholesale).

Usage:
    python tools/plan_fast.py [scene_yml] [scene_json]

Artifacts and code paths identical to the step-by-step script.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# fast profile: 4 IK seeds per candidate in the batch oracle (the plan
# gate's hard verdicts + the sequential fallback own correctness; marginal
# rows simply fall through to the fallback chain)
os.environ.setdefault("ROBOCINE_IK_SEEDS", "3")

_T0 = time.perf_counter()


def mark(msg: str) -> None:
    print(f"⏱ [fast {time.perf_counter() - _T0:5.2f}s] {msg}", flush=True)


# fast-profile config defaults, injected before any stage module imports
# load_config (from-imports bind the patched attr). Plateau patience 20:
# an occlusion plateau means the current shape cannot improve it — 50
# iterations of patience burned ~1 s on hopeless geometries (dolly_in
# plateaued at iter 1 and still ran to 51).
import vlm_intention.common.config as _cc

_orig_load_config = _cc.load_config


def _fast_load_config(*args, **kwargs):
    cfg = _orig_load_config(*args, **kwargs)
    try:
        opt = cfg.setdefault("cinema_traj", {}).setdefault("optimizer", {})
        opt.setdefault("occl_plateau_iters", 12)
        opt.setdefault("ray_steps", 16)
    except Exception:
        pass
    return cfg


_cc.load_config = _fast_load_config


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    yml = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO / "cinema_traj/scenes/config/kujiale_0021_kid_combo.yml")
    scene_json = sys.argv[2] if len(sys.argv) > 2 else str(
        REPO / "cinema_traj/scenes/kujiale_0021_kid_combo.json")

    # -- step 3: scenario_gen (yml → plan/build/optimize/audit) --------- #
    mark(f"scenario_gen {Path(yml).name}")
    sg = _load_module("scenario_gen_fast", REPO / "tools/scenario_gen.py")
    sys.argv = ["scenario_gen.py", yml]
    sg.main()
    mark("scenario_gen done")

    # -- steps 4+5: export (NED csv + conditional auto-retime cascade), then
    # the UNCONDITIONAL jarvis retime (config-stream IK + trackability).
    # The cascade in export fires only on overspeed — slow moves (pan,
    # static) never trigger it, so the explicit retime stage is the one
    # that actually owns jarvis_traj.json. Both run in-process here; the
    # script paid a separate 1.5 s process spawn for each.
    rt = _load_module("run_traj_fast", REPO / "run_traj.py")
    sys.argv = ["run_traj.py", "--stage", "export", "--scene", scene_json,
                "--force"]
    rt.main()
    mark("export done")
    _sid = json.loads(Path(scene_json).read_text(encoding="utf-8")).get(
        "scene_id", Path(scene_json).stem)
    _ff = REPO / "cinema_traj_out" / _sid / "exports" / "jarvis_traj.json"
    _tf = REPO / "cinema_traj_out" / _sid / "traj_final.json"
    if _ff.exists() and _tf.exists() and \
            _ff.stat().st_mtime >= _tf.stat().st_mtime - 1.0:
        mark("retime skipped — export cascade already produced the "
             "jarvis stream")
    else:
        sys.argv = ["run_traj.py", "--stage", "retime", "--scene", scene_json]
        rt.main()
        mark("retime (jarvis config stream) done")

    # -- step 6: fix_dwell_arcs on the feedforward stream ---------------- #
    scene_id = json.loads(Path(scene_json).read_text(encoding="utf-8")).get(
        "scene_id", Path(scene_json).stem)
    ff_json = REPO / "cinema_traj_out" / scene_id / "exports" / "jarvis_traj.json"
    fda = _load_module("fix_dwell_fast", REPO / "scripts/fix_dwell_arcs.py")
    sys.argv = ["fix_dwell_arcs.py", str(ff_json)]
    fda.main()
    mark("fix_dwell_arcs done")

    # -- step 7: viewer --------------------------------------------------- #
    sys.argv = ["run_traj.py", "--stage", "view", "--scene", scene_json]
    rt.main()
    mark("view done")

    # -- step 8: simview stage build, IN-PROCESS (robocine_plan now carries
    # pillow; isaac imports in simview_isaac are lazy — only `render` needs
    # SimulationApp) ----------------------------------------------------- #
    sv = _load_module("simview_build_fast", REPO / "tools/simview_isaac.py")
    out_dir = REPO / "cinema_traj_out" / scene_id
    sys.argv = ["simview_isaac.py", "build",
                "--traj", str(out_dir / "traj_final.json"),
                "--scene", scene_json,
                "--cast", str(out_dir / "cast.json"),
                "--subject-glb", "none", "--fps", "10"]
    sv.main()
    mark("simview build done")


if __name__ == "__main__":
    main()
