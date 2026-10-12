"""USD-only startup geometry checks (no SimulationApp or GPU required)."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pxr.Usd")
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def load_runner():
    path = (Path(__file__).resolve().parents[1] / "ros2_ws/jvs_mock/scripts"
            / "jvs_isaac_sim.py")
    spec = importlib.util.spec_from_file_location("jvs_spawn_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("position,yaw", [
    ((0.0, 0.0, 0.002), 0.0),
    ((5.0, -1.0, 0.002), -math.pi / 2),
    ((-2.0, 3.0, 0.01), 0.7),
])
def test_spawn_preserves_joint_anchors_and_link_frames(position, yaw):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/robot")
    scene = UsdGeom.Xform.Define(stage, "/scene")
    scene.AddTranslateOp().Set(Gf.Vec3d(9, 8, 7))

    # Match the importer's flat layout and double-precision rest transforms.
    offsets = {
        "base_link": (0.0, 0.0, 0.0),
        "wheel": (0.0, 0.105, 0.07),
        "arm": (0.098, 0.0, 0.266),
    }
    for name, offset in offsets.items():
        body = UsdGeom.Xform.Define(stage, f"/robot/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*offset))
        body.AddOrientOp().Set(Gf.Quatf(1.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        if name != "base_link":
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/robot/{name}_joint")
            joint.CreateBody0Rel().SetTargets(["/robot/base_link"])
            joint.CreateBody1Rel().SetTargets([f"/robot/{name}"])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))

    camera = UsdGeom.Xform.Define(stage, "/robot/arm/camera")
    camera.AddTranslateOp().Set(Gf.Vec3d(0.02, 0, 0.1))
    camera.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 90))
    cache = UsdGeom.XformCache()
    rest = {str(p.GetPath()): cache.GetLocalToWorldTransform(p)
            for p in stage.Traverse() if p.IsA(UsdGeom.Xform)}
    args = SimpleNamespace(spawn_x=position[0], spawn_y=position[1],
                           spawn_z=position[2], spawn_yaw=yaw)
    runner = load_runner()
    # Reconfiguration must set an absolute spawn, not accumulate transforms.
    for _ in range(2):
        runner.configure_robot_spawn(stage, "/robot/base_link", args)
        cache.Clear()
        spawn = Gf.Matrix4d(1.0)
        spawn.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
        spawn.SetTranslateOnly(Gf.Vec3d(*position))
        for path in ("/robot/base_link", "/robot/wheel", "/robot/arm", "/robot/arm/camera"):
            actual = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            np.testing.assert_allclose(actual, rest[path] * spawn, atol=1e-6)
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            a = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(
                joint.GetBody0Rel().GetTargets()[0])).Transform(joint.GetLocalPos0Attr().Get())
            b = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(
                joint.GetBody1Rel().GetTargets()[0])).Transform(joint.GetLocalPos1Attr().Get())
            assert (a - b).GetLength() < 1e-6
        assert cache.GetLocalToWorldTransform(scene.GetPrim()) == rest["/scene"]
    for op in UsdGeom.Xformable(stage.GetPrimAtPath("/robot/base_link")).GetOrderedXformOps():
        assert op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
