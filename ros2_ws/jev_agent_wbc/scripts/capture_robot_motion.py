#!/usr/bin/env python3
"""Read-only PhysX link-pose capture around the deployed validation entrypoints.

Run with the selected robot's Python. No rendering, controller changes, state
writes or physics-frequency changes are introduced by these capture hooks.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "g1": ROOT / "Robot_G1/scripts/validate_locomotion.py",
    "go2": ROOT / "Robot_Go2/scripts/validate_wbc.py",
    "ridgeback": ROOT / "Robot_Ridgeback_UR5/scripts/run_isaac.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = TARGETS[args.robot]
    sys.path.insert(0, str(target.parent))
    if args.robot == "ridgeback":
        sys.path.insert(0, str(target.parent.parent))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    source_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "validator_source.py").write_bytes(target.read_bytes())
    import isaacsim
    original_app = isaacsim.SimulationApp

    class Capture:
        def __init__(self):
            self.frames = []
            self.records = []
            self.paths = None
            self.finished = False

        def source_frame(self):
            frame = inspect.currentframe().f_back
            while frame:
                if frame.f_code.co_filename == str(target):
                    return frame
                frame = frame.f_back
            return None

        def append(self, view, names, t, phase, extras):
            import numpy as np
            import omni.usd
            raw = view.get_link_transforms()
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            poses = np.asarray(raw)[0].copy()
            # PhysX tensors are xyz + xyzw; the artifact uses xyz + wxyz.
            poses = poses[:, [0, 1, 2, 6, 3, 4, 5]].astype(np.float32)
            if not np.isfinite(poses).all():
                raise FloatingPointError("Nonfinite physical link pose")
            if self.paths is None:
                self.paths = list(view.link_paths[0])
                self.names = list(names)
                if len(self.paths) != len(poses):
                    raise ValueError("PhysX link paths and transforms differ")
                omni.usd.get_context().get_stage().GetRootLayer().Export(str(output / "scene.usda"))
                print("CAPTURE_LINKS", len(self.paths), self.paths, flush=True)
            self.frames.append(poses)
            self.records.append(dict(time=float(t), phase=str(phase), **extras))

        def finalize(self):
            if self.finished:
                return
            self.finished = True
            import numpy as np
            if not self.frames:
                (output / "capture_error.txt").write_text("No physical frames captured\n")
                return
            np.savez_compressed(output / "measured_links.npz", poses=np.stack(self.frames),
                                times=np.array([r["time"] for r in self.records]))
            with (output / "frames.jsonl").open("w") as f:
                for row in self.records:
                    f.write(json.dumps(row) + "\n")
            meta = dict(protocol="measured_link_poses/v1", measured=True, robot=args.robot,
                        source=str(target), source_sha256=source_hash,
                        capture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        pose_convention="world xyz + quaternion wxyz", body_paths=self.paths,
                        body_names=self.names, frames=len(self.frames),
                        physical_duration_s=self.records[-1]["time"], live_rgb=False,
                        control_state_writes_by_capture=0, physics_steps_by_capture=0)
            (output / "capture.json").write_text(json.dumps(meta, indent=2) + "\n")
            print("CAPTURE_COMPLETE", json.dumps(meta), flush=True)

        def install(self):
            if args.robot == "g1":
                from isaaclab.assets import Articulation
                old_update = Articulation.update
                capture = self

                def update(robot, dt):
                    result = old_update(robot, dt)
                    frame = capture.source_frame()
                    if frame and frame.f_code.co_name == "main":
                        local = frame.f_locals
                        if "t" in local and "decimation" in local and local.get("_") == local["decimation"] - 1:
                            capture.append(robot.root_physx_view, robot.body_names, local["t"] + local["dt"],
                                           local["c"]["name"], {"command": local["cmd"][0].tolist()})
                    return result
                Articulation.update = update
            elif args.robot == "go2":
                from isaaclab.envs import ManagerBasedRLEnv
                old_step = ManagerBasedRLEnv.step
                capture = self

                def step(env, action):
                    result = old_step(env, action)
                    frame = capture.source_frame()
                    if frame and "case" in frame.f_locals:
                        local = frame.f_locals
                        robot = env.scene["robot"]
                        t = len(capture.records) * env.step_dt + env.step_dt
                        capture.append(robot.root_physx_view, robot.body_names, t, local["case"],
                                       {"case_time": float(local["t"] + env.step_dt), "command": list(local["command"])})
                    return result
                ManagerBasedRLEnv.step = step
            else:
                from isaacsim.core.api import World
                old_step = World.step
                capture = self

                def step(world, *pos, **kw):
                    result = old_step(world, *pos, **kw)
                    frame = capture.source_frame()
                    if frame and frame.f_code.co_name == "step" and "phase" in frame.f_locals:
                        local = frame.f_locals
                        robot = world.scene.get_object("ridgeback_ur5")
                        view = robot._articulation_view
                        capture.append(view._physics_view, view.body_names, (local["steps"] + 1) * .01,
                                       local["phase"], {})
                    return result
                World.step = step

    capture = Capture()

    class CapturedApp(original_app):
        def __init__(self, *pos, **kw):
            super().__init__(*pos, **kw)
            capture.install()

        def close(self, *pos, **kw):
            capture.finalize()
            return super().close(*pos, **kw)

    isaacsim.SimulationApp = CapturedApp
    validator_args = [str(target), "--output", str(output / "physical")]
    if args.seconds is not None:
        if args.robot == "ridgeback":
            validator_args.append("--probe")
        else:
            validator_args += ["--seconds", str(args.seconds)]
    sys.argv = validator_args
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        capture.finalize()


if __name__ == "__main__":
    main()
