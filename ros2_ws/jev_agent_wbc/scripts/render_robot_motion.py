#!/usr/bin/env python3
"""Render measured robot link poses at 1x; no physics or policy is executed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PHASES = {"stand":"站立", "settle":"初始稳定", "forward":"前进", "backward":"后退",
          "left":"左侧移", "right":"右侧移", "diagonal":"斜向行走",
          "diagonal_left":"左前斜行", "diagonal_right":"右前斜行", "diagonal_ne":"左前斜行",
          "diagonal_sw":"右后斜行", "yaw_left":"左转", "yaw_right":"右转",
          "yaw_ccw":"逆时针转向", "yaw_cw":"顺时针转向", "stop":"停止",
          "pre_wbc":"协调控制准备", "coordinated":"行走＋机械臂协调", "whole_body":"底盘＋UR5全身协调",
          "final_stop":"最终停止"}
TITLES = {"g1":"Unitree G1  ·  AGILE WBC", "go2":"Unitree Go2 ＋ Piper  ·  LeggedManip WBC",
          "ridgeback":"Clearpath Ridgeback ＋ UR5"}
NOTES = {"g1":"29自由度模型｜多方向连续行走｜侧移速度存在跟踪偏差",
         "go2":"12腿关节＋6臂关节｜自由浮动底座｜工况之间重置",
         "ridgeback":"官方 x / y / yaw 平面关节底盘｜本片不验证轮地摩擦"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    folder = args.capture.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "renderer_source.py").write_bytes(Path(__file__).read_bytes())
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=True, width=1280, height=600, renderer="RaytracedLighting",
                             anti_aliasing=0, multi_gpu=False, limit_cpu_threads=8))
    encoder = None
    try:
        import numpy as np
        import carb
        import omni.usd
        import omni.physx
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdLux
        from scipy.spatial.transform import Rotation
        from PIL import Image, ImageDraw, ImageFont

        meta = json.loads((folder / "capture.json").read_text())
        if meta["protocol"] != "measured_link_poses/v1" or not meta["measured"]:
            raise ValueError("Requires measured physical link poses")
        robot = meta["robot"]
        data = np.load(folder / "measured_links.npz", allow_pickle=False)
        poses, times = data["poses"], data["times"]
        records = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
        assert poses.shape == (len(records), len(meta["body_paths"]), 7)
        assert np.all(np.diff(times) > 0) and np.isfinite(poses).all()
        assert np.max(np.abs(np.array([r["time"] for r in records]) - times)) < 1e-9
        duration = float(times[-1])
        desired_times = np.arange(1, round(duration * args.fps) + 1) / args.fps
        indices = np.searchsorted(times, desired_times - 1e-8)
        if args.limit:
            indices = indices[:args.limit]
        context = omni.usd.get_context()
        context.open_stage(str(folder / "scene.usda"))
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        omni.timeline.get_timeline_interface().stop()
        settings = carb.settings.get_settings()
        settings.set("/app/player/playSimulations", False)
        settings.set("/rtx/raytracing/cached/enabled", False)
        physics_events = []
        subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            lambda dt: physics_events.append(float(dt)))
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

        operations = []
        for path in meta["body_paths"]:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Missing physical body in stage: " + path)
            xf = UsdGeom.Xformable(prim)
            operation = xf.MakeMatrixXform()
            xf.SetResetXformStack(True)
            operations.append(operation)
        names = meta["body_names"]
        base_id = names.index("base_link") if robot == "ridgeback" else 0
        base_track = poses[:, base_id, :3]
        floor_z = -.0289 if robot == "ridgeback" else 0.

        def curves(path, points, counts, color, width):
            curve = UsdGeom.BasisCurves.Define(stage, path)
            curve.CreateTypeAttr("linear")
            curve.CreatePointsAttr(points)
            curve.CreateCurveVertexCountsAttr(counts)
            curve.CreateWidthsAttr([width])
            curve.SetWidthsInterpolation("constant")
            curve.CreateDisplayColorAttr([color])
            return curve

        grid = []
        for value in np.arange(-8, 8.01, .5):
            grid.extend([(-8., float(value), floor_z+.004), (8., float(value), floor_z+.004),
                         (float(value), -8., floor_z+.004), (float(value), 8., floor_z+.004)])
        curves("/Review/Grid", grid, [2] * (len(grid)//2), (.48,.52,.57), .008)
        curves("/Review/X", [(0,0,floor_z+.012),(1,0,floor_z+.012)], [2], (.9,.16,.12), .025)
        curves("/Review/Y", [(0,0,floor_z+.012),(0,1,floor_z+.012)], [2], (.1,.8,.3), .025)
        trail = curves("/Review/MeasuredBaseTrail", [(0,0,floor_z+.01),(0,0,floor_z+.01)], [2], (.1,.85,.6), .016)
        camera = UsdGeom.Camera.Define(stage, "/Review/Camera")
        camera.CreateFocalLengthAttr(36.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(36.*600/1280)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 1000.))
        camera_op = camera.MakeMatrixXform()
        product = rep.create.render_product(str(camera.GetPath()), (1280,600))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(product)
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        font = ImageFont.truetype(font_path, 27)
        small = ImageFont.truetype(font_path, 21)
        tiny = ImageFont.truetype(font_path, 17)
        target_file = output / (robot + ".mp4")
        encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target_file)], stdin=subprocess.PIPE)
        max_position = max_rotation = 0.
        frame_index = []
        phase_start = 0
        previous_phase = None
        started = time.monotonic()
        for n, index in enumerate(indices):
            row = records[int(index)]
            current_poses = poses[index]
            for op, pose in zip(operations, current_poses):
                q = pose[3:].astype(float)
                q /= np.linalg.norm(q)
                matrix = Gf.Matrix4d(1.)
                matrix.SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float,q[1:]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
                op.Set(matrix)
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            for path, pose in zip(meta["body_paths"], current_poses):
                matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                position = np.array(matrix.ExtractTranslation())
                quat = matrix.ExtractRotationQuat()
                qxyzw = np.r_[np.array(quat.GetImaginary()), float(quat.GetReal())]
                max_position = max(max_position, float(np.linalg.norm(position-pose[:3])))
                expected = Rotation.from_quat(pose[[4,5,6,3]])
                max_rotation = max(max_rotation, float((Rotation.from_quat(qxyzw) * expected.inv()).magnitude()))
            bp = base_track[index].astype(float)
            if robot == "g1":
                eye, target = bp + [2.4,2.4,1.25], bp + [0.,0.,-.03]
            elif robot == "ridgeback":
                eye, target = bp + [2.4,2.4,1.6], bp + [.05,0.,.35]
            else:
                eye, target = bp + [1.8,1.8,1.15], bp + [0.,0.,.2]
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1))
            camera_op.Set(view.GetInverse())
            if row["phase"] != previous_phase:
                if robot == "go2":
                    phase_start = int(index)
                previous_phase = row["phase"]
            history = base_track[phase_start:int(index)+1:5].copy()
            if len(history) < 2:
                history = np.repeat(history,2,axis=0)
            history[:,2] = floor_z+.015
            trail.GetPointsAttr().Set([Gf.Vec3f(*map(float,p)) for p in history])
            trail.GetCurveVertexCountsAttr().Set([len(history)])
            settings.set("/app/player/playSimulations", False)
            rep.orchestrator.step(rt_subframes=1, pause_timeline=True, delta_time=0.0)
            pixels = np.asarray(rgb.get_data())
            if pixels.shape[:2] != (600,1280):
                raise ValueError(f"Unexpected RGB image shape {pixels.shape}")
            canvas = Image.new("RGB", (1280,720), (14,20,31))
            canvas.paste(Image.fromarray(pixels[:,:,:3]), (0,64))
            draw = ImageDraw.Draw(canvas)
            draw.text((22,12), TITLES[robot], font=font, fill=(239,246,255))
            label = PHASES.get(row["phase"],row["phase"])
            draw.text((835,16), f"{label}   {row['time']:05.2f}s", font=small, fill=(104,229,194))
            draw.text((20,670), NOTES[robot], font=small, fill=(225,231,240))
            draw.text((20,643), "实测状态回放 · 原速 1× · 未接入避障Agent", font=tiny, fill=(242,246,250),
                      stroke_width=2,stroke_fill=(20,25,33))
            if "command" in row:
                cmd=row["command"]
                draw.text((20,80), f"指令 vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f} m/s   wz={cmd[2]:+.2f} rad/s",
                          font=tiny, fill=(255,255,255),stroke_width=2,stroke_fill=(22,27,33))
            # Measured XY trace makes translations visible with the following camera.
            x0,y0,x1,y1 = 1050,455,1250,635
            draw.rounded_rectangle((x0,y0,x1,y1),radius=10,fill=(20,29,40))
            draw.text((x0+12,y0+7),"实测基座轨迹 XY",font=tiny,fill=(220,232,244))
            trace=base_track[phase_start:int(index)+1:5,:2]
            origin=base_track[phase_start,:2]
            extent=max(.8,float(np.max(np.abs(trace-origin)))+.2)
            mapped=[(float(x0+100+(p[0]-origin[0])*70/extent),float(y0+110-(p[1]-origin[1])*65/extent)) for p in trace]
            if len(mapped)>1:draw.line(mapped,fill=(70,221,170),width=3)
            px,py=mapped[-1];draw.ellipse((px-4,py-4,px+4,py+4),fill=(255,212,94))
            draw.text((x0+12,y1-27),f"范围 ±{extent:.1f}m",font=tiny,fill=(165,183,204))
            draw.rectangle((0,716,int(1280*(n+1)/len(indices)),719),fill=(60,210,165))
            encoder.stdin.write(canvas.tobytes())
            if n in (0,len(indices)//2,len(indices)-1):
                canvas.save(output/f"frame_{n:05d}.jpg",quality=94)
            frame_index.append(dict(frame=n,video_time=n/args.fps,physical_time=float(times[index]),source_index=int(index),phase=row["phase"]))
            if n%100==0:print("RENDER",robot,n,"/",len(indices),flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoder failed")
        encoder = None
        (output/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frame_index))
        audit=dict(robot=robot,frame_count=len(indices),fps=args.fps,video_duration_s=len(indices)/args.fps,
                   measured_state_replay=True,live_sensor_capture=False,physics_steps_during_render=len(physics_events),
                   max_link_position_error_m=max_position,max_link_rotation_error_rad=max_rotation,
                   physical_source_sha256=hashlib.sha256((folder/"measured_links.npz").read_bytes()).hexdigest(),
                   scene_sha256=hashlib.sha256((folder/"scene.usda").read_bytes()).hexdigest(),
                   renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   render_wall_s=time.monotonic()-started,
                   valid=bool(max_position<1e-7 and max_rotation<1e-6 and not physics_events))
        (output/"replay_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
        print("REPLAY_AUDIT",json.dumps(audit),flush=True)
        if not audit["valid"]:raise RuntimeError("Replay pose audit failed")
        rgb.detach(product)
        product.destroy()
        del subscription
    finally:
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
