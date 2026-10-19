#!/usr/bin/env python3
"""Import the assembled Go2+D1 URDF to an unfixed Isaac Sim articulation."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=False)
faulthandler.dump_traceback_later(45, repeat=True)

ROOT = Path(__file__).resolve().parents[1]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu", headless=True)
args = parser.parse_args()
print("D1_APP_LAUNCH_BEGIN", flush=True)
launcher = AppLauncher(args)
app = launcher.app
print("D1_APP_LAUNCH_DONE", flush=True)
try:
    import omni.kit.app
    import omni.kit.commands
    from pxr import Usd, UsdPhysics
    manager = omni.kit.app.get_app().get_extension_manager()
    # The fixed Lab main converter requests 2.4.31 from Nucleus. Sim 5.1 ships
    # 2.4.30 locally; use its documented native API and preserve the original URDF.
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf-2.4.30", True)
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Could not create native URDF import configuration")
    cfg.set_distance_scale(1.0)
    cfg.set_make_default_prim(True)
    cfg.set_create_physics_scene(False)
    cfg.set_merge_fixed_joints(True)
    cfg.set_fix_base(False)
    cfg.set_self_collision(False)
    cfg.set_import_inertia_tensor(True)
    output = ROOT / "assets/generated/go2_d1/go2_d1.usd"
    output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / "assets/generated/go2_d1.urdf"
    print("D1_URDF_CONVERSION_BEGIN", flush=True)
    status, robot_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(source), import_config=cfg, dest_path=str(output))
    if not status or not output.is_file():
        raise RuntimeError("Native URDF import did not produce a USD asset")
    from clean_empty_visual_refs import clean_empty_head_visual_references
    cleaned = clean_empty_head_visual_references(output.parent / "configuration/go2_d1_base.usd")
    stage = Usd.Stage.Open(str(output))
    joints = [str(prim.GetPath()) for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)]
    report = {"output": str(output), "importer": "isaacsim.asset.importer.urdf-2.4.30",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "usd_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "default_prim": str(stage.GetDefaultPrim().GetPath()), "movable_joints": joints,
              "movable_joint_count": len(joints), "fix_base": False,
              "removed_proven_dangling_head_visual_references": cleaned,
              "layer_sha256": {str(p.relative_to(output.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in output.parent.rglob("*.usd")},
              "conversion_pass": len(joints) == 20,
              "dynamics_validated": False,
              "limitation": "Original D1 arm/finger effort and velocity limits are zero. No Piper policy was applied."}
    (output.parent / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("IMPORTED " + str(output), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if not report["conversion_pass"]:
        raise RuntimeError("Expected 12 leg + 6 arm + 2 finger joints")
finally:
    faulthandler.cancel_dump_traceback_later()
    from isaaclab.sim import SimulationContext
    context = SimulationContext.instance()
    if context is not None:
        context._disable_app_control_on_stop_handle = True
    app.close(wait_for_replicator=False)
