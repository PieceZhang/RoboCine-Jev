#!/usr/bin/env python3
"""Render synchronized optical/third-person views of audited Isaac demo telemetry.

This is a visual replay of measured states, not a fresh controller experiment.
All cameras render the same USD state with the simulation clock paused. Saved
physical elapsed time, rather than the adaptive camera-path clock, drives replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

DEMOS = {
    "crossing": ("机械臂横向柱避障", "combo1"),
    "overtake": ("底盘未来路线横穿", "combo3"),
    "two_movers": ("双障碍临时阻挡", "combo4"),
}
HEADER_HEIGHT = 48
FOOTER_HEIGHT = 76
RECORDING_RED = (255, 48, 62)
IDLE_GREEN = (40, 221, 126)
SHOT_TYPES = ("orbit", "pan", "moveout")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(folder):
    with (folder / "telemetry.csv").open() as source:
        rows = list(csv.DictReader(source))
    elapsed = np.array([float(row["wall_elapsed_s"]) for row in rows])
    if not np.all(np.diff(elapsed) > 0):
        raise ValueError(f"non-monotonic physical telemetry: {folder}")
    columns = ["state_x", "state_y", "base_z", *[f"state_q{i}" for i in range(1, 7)]]
    states = np.array([[float(row[key]) for key in columns] for row in rows])
    rotations = Rotation.from_rotvec(
        np.array([[float(row[f"base_rotvec_{axis}"]) for axis in "xyz"] for row in rows])
    )
    # Wheel velocities are recorded; integrate them only for visual wheel spin.
    wheel_rates = np.array([[float(row[f"wheel_{i}"]) for i in range(2)] for row in rows])
    wheel_angles = np.vstack([np.zeros(2), np.cumsum(
        .5 * (wheel_rates[1:] + wheel_rates[:-1]) * np.diff(elapsed)[:, None], axis=0)])
    states = np.column_stack([states, wheel_angles])
    trajectory = json.loads((folder / "trajectory.json").read_text())
    reference = json.loads((folder / "reference.json").read_text())
    spawn = trajectory["initial_base_pose"]
    origin = np.array([spawn["x"], -spawn["z"], spawn["height"]])
    initial_rotation = Rotation.from_euler("z", spawn["yaw"] - math.pi / 2)
    with (folder / "obstacle_truth.csv").open() as source:
        truth = list(csv.DictReader(source))
    obstacles = json.loads((folder / "obstacle_scene.json").read_text())["obstacles"]
    mover_samples = {}
    for obstacle in obstacles:
        samples = [row for row in truth if row["obstacle"] == obstacle["name"]]
        # Before the epoch the truth log repeatedly reports scene time zero.
        times = np.array([float(row["scene_time_s"]) for row in samples])
        times, indices = np.unique(times, return_index=True)
        positions = np.array([[float(row[key]) for key in "xyz"] for row in samples])[indices]
        mover_samples[obstacle["name"]] = (times, positions)
    camera_positions = np.array([[float(row[f"camera_{axis}"]) for axis in "xyz"] for row in rows])
    return SimpleNamespace(rows=rows, elapsed=elapsed, states=states, camera_positions=camera_positions,
        slerp=Slerp(elapsed, rotations),
        spawn=spawn, origin=origin, initial_rotation=initial_rotation,
        obstacles=obstacles, movers=mover_samples, reference=reference,
        duration=float(elapsed[-1]))


class RobotReplay:
    """URDF link transforms from measured base pose and six measured joint angles."""

    def __init__(self, stage, robot_path, urdf):
        from pxr import UsdGeom, UsdPhysics
        import jvs_dynamic_simview_isaac as runner

        document = ET.parse(urdf).getroot()
        links = [link.attrib["name"] for link in document.findall("link")]
        self.ops = {}
        for name in links:
            prim = runner.find_robot_prim(stage, robot_path, name)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            for key in tuple(prim.GetPropertyNames()):
                if key.startswith("xformOp:"):
                    prim.RemoveProperty(key)
            xform.SetResetXformStack(True)
            self.ops[name] = (xform.AddTranslateOp(), xform.AddOrientOp())
        # Rendering never starts physics; disable imported dynamics explicitly.
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        self.children = {}
        child_links = set()
        for joint in document.findall("joint"):
            parent, child = [joint.find(key).attrib["link"] for key in ("parent", "child")]
            child_links.add(child)
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz") if axis is not None else "1 0 0", sep=" ")
            self.children.setdefault(parent, []).append(
                (child, joint.attrib["name"], xyz, Rotation.from_euler("xyz", rpy), axis,
                 joint.attrib["type"] != "fixed"))
        self.root = (set(links) - child_links).pop()

    def update(self, run, t):
        from pxr import Gf

        t = float(np.clip(t, run.elapsed[0], run.elapsed[-1]))
        state = np.array([np.interp(t, run.elapsed, run.states[:, j]) for j in range(run.states.shape[1])])
        base_position = run.origin + run.initial_rotation.apply(state[:3])
        base_rotation = run.initial_rotation * run.slerp(t)
        angles = {f"joint{i}": state[i + 2] for i in range(1, 7)}
        angles.update(left_wheel_joint=state[9], right_wheel_joint=state[10])
        poses = {}

        def visit(name, position, rotation):
            poses[name] = (position, rotation)
            translate, orient = self.ops[name]
            translate.Set(Gf.Vec3d(*position))
            quat = rotation.as_quat()
            orient.Set(Gf.Quatf(float(quat[3]), Gf.Vec3f(*quat[:3])))
            for child, joint, xyz, origin_rotation, axis, moving in self.children.get(name, []):
                child_rotation = rotation * origin_rotation
                if moving:
                    child_rotation = child_rotation * Rotation.from_rotvec(axis * angles.get(joint, 0.))
                visit(child, position + rotation.apply(xyz), child_rotation)

        visit(self.root, base_position, base_rotation)
        # Verify the reconstructed optical mounting against recorded camera FK.
        expected = run.origin + run.initial_rotation.apply(
            [np.interp(t, run.elapsed, run.camera_positions[:, j]) for j in range(3)])
        error = float(np.linalg.norm(poses["camera_link"][0] - expected))
        return error


def create_movers(stage, run):
    from pxr import Gf, UsdGeom

    ops = {}
    q = run.initial_rotation.as_quat()
    for obstacle in run.obstacles:
        model, name = obstacle["model"], obstacle["name"]
        root = UsdGeom.Xform.Define(stage, "/World/DynamicObstacles/" + name)
        ops[name] = root.AddTranslateOp()
        root.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
        path = root.GetPath().AppendChild("Model")
        if model["type"] == "box":
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.)
            shape.AddScaleOp().Set(Gf.Vec3f(*model["size"]))
        elif model["type"] == "cylinder":
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateAxisAttr(model.get("axis", "z").upper())
            shape.CreateRadiusAttr(model["radius"])
            shape.CreateHeightAttr(model["height"])
        else:
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(model["radius"])
        shape.CreateDisplayColorAttr([Gf.Vec3f(*model["color"])])
    return ops


def update_movers(run, ops, t):
    from pxr import Gf
    for name, op in ops.items():
        times, positions = run.movers[name]
        local = np.array([np.interp(t, times, positions[:, j]) for j in range(3)])
        op.Set(Gf.Vec3d(*(run.origin + run.initial_rotation.apply(local))))


def create_environment_camera(stage, run):
    from pxr import Gf, UsdGeom

    points = np.vstack([run.states[:, :2], *[p[:, :2] for _, p in run.movers.values()]])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    target = run.origin + run.initial_rotation.apply([*center, .40])
    # Stay below the apartment ceiling and clear of wall-mounted fixtures.
    eye = target + run.initial_rotation.apply([-.35, -.35, 2.00])
    camera = UsdGeom.Camera.Define(stage, "/World/DualViewEnvironmentCamera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 500.))
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateFocalLengthAttr(18.)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    camera.AddTransformOp().Set(view.GetInverse())
    return str(camera.GetPath()), dict(position=eye.tolist(), target=target.tolist(), focal_length_mm=18.)


def create_side_cameras(stage, run):
    """Fixed left/right observers relative to the initial base heading."""
    from pxr import Gf, UsdGeom

    points = run.states[:, :2]
    center = .5 * (points.min(axis=0) + points.max(axis=0))
    target = run.origin + run.initial_rotation.apply([*center, .45])
    cameras, poses = [], {}
    for side, sign in (("left", 1.), ("right", -1.)):
        eye = target + run.initial_rotation.apply([-.20, sign * 1.15, 1.00])
        camera = UsdGeom.Camera.Define(stage, f"/World/QuadView_{side.title()}Camera")
        camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 500.))
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateFocalLengthAttr(16.)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        camera.AddTransformOp().Set(view.GetInverse())
        cameras.append(str(camera.GetPath()))
        poses[side] = dict(position=eye.tolist(), target=target.tolist(), focal_length_mm=16.)
    return cameras, poses


@lru_cache(maxsize=4)
def caption_font(size):
    from PIL import ImageFont
    return ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size)


def frame_annotation(run, t):
    """Use only a telemetry sample already reached by the physical replay clock.

    The path clock can slow during avoidance. It is used only to cross-check
    the recorded transition against the execution reference, never to advance
    video frames or anticipate the next phase. Arc names inter-shot whole-body
    reconfiguration; the camera may hold its pose while the base repositions.
    Initial focus acquisition and the final hold are idle, not an arc.
    """
    index = int(np.searchsorted(run.elapsed, t, side="right")) - 1
    if index < 0:
        return dict(sample_index=None, sample_wall_elapsed_s=None, phase="idle",
                    label="idle · 准备", recording=False,
                    position_error_cm=None, orientation_error_deg=None)
    index = min(index, len(run.rows) - 1)
    row = run.rows[index]
    phase = row["phase"]
    shot = row["shot_type"]
    path_time = float(row["time_s"])
    transition_active = bool(int(row["transition_active"]))
    motion_active = bool(int(row["motion_active"]))
    # Require both a true inter-shot window in the saved execution reference
    # and its controller transition flag. motion_active=0 by itself also
    # describes focus acquisition and final holds, which must remain idle.
    shots = run.reference.get("execution_shots", [])
    connector = any(float(previous.get("motion_end_s", previous["end_s"])) <= path_time
                    < float(following.get("motion_start_s", following["start_s"]))
                    for previous, following in zip(shots, shots[1:]))
    arc = phase == "tracking" and transition_active and connector
    recording = (phase == "tracking" and motion_active and not transition_active
                 and shot in SHOT_TYPES)
    if arc:
        label, display_phase = "arc · 衔接阶段", "arc"
    elif recording:
        label, display_phase = f"{shot} · 运镜拍摄", shot
    else:
        label, display_phase = "idle · 空闲", "idle"
        if phase == "preparing":
            label = "idle · 准备"
        elif phase == "stopping":
            label = "idle · 录制结束"
    return dict(sample_index=index, sample_wall_elapsed_s=float(run.elapsed[index]),
                phase=display_phase, label=label, recording=recording,
                position_error_cm=float(row["position_error"]) * 100. if recording else None,
                orientation_error_deg=math.degrees(float(row["orientation_error"])) if recording else None)


def draw_path_inset(draw, run, t, rectangle):
    """Show measured base/actor trails up to this physical timestamp only."""
    x0, y0, x1, y1 = rectangle
    robot_end = int(np.searchsorted(run.elapsed, t, side="right"))
    paths = [(run.states[:robot_end, :2], (255, 215, 70))]
    for times, positions in run.movers.values():
        end = int(np.searchsorted(times, t, side="right"))
        paths.append((positions[:end, :2], (60, 170, 255)))
    measured = [points for points, _ in paths if len(points)]
    draw.rounded_rectangle(rectangle, radius=10, fill=(19, 30, 43),
                           outline=(126, 150, 177), width=1)
    draw.text((x0 + 12, y0 + 8), "实测路径 · 俯视", font=caption_font(18), fill=(232, 241, 252))
    draw.text((x0 + 12, y1 - 29), "黄：底盘   蓝：障碍物", font=caption_font(16), fill=(220, 232, 248))
    if not measured:
        return
    points = np.vstack(measured)
    center = .5 * (points.min(axis=0) + points.max(axis=0))
    span = max(float(np.ptp(points, axis=0).max()) + .25, .8)
    size = min(x1 - x0 - 30, y1 - y0 - 73)
    plot_center = np.array([.5 * (x0 + x1), .5 * (y0 + 37 + y1 - 36)])
    for trail, color in paths:
        if not len(trail):
            continue
        pixels = plot_center + (trail - center) * np.array([1., -1.]) * size / span
        if len(pixels) > 1:
            draw.line([tuple(point) for point in pixels], fill=color, width=3)
        x, y = pixels[-1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=(255, 255, 255), width=1)


def caption_frame(left, right, name, run, t, path_inset=False, side_views=None):
    from PIL import Image, ImageDraw

    height, width = left.shape[:2]
    rows = 2 if side_views is not None else 1
    footer_y = rows * (height + HEADER_HEIGHT)
    frame = Image.new("RGB", (width * 2, footer_y + FOOTER_HEIGHT), (19, 26, 35))
    frame.paste(Image.fromarray(left), (0, HEADER_HEIGHT))
    frame.paste(Image.fromarray(right), (width, HEADER_HEIGHT))
    draw = ImageDraw.Draw(frame)
    font = caption_font(22)
    small = caption_font(18)
    annotation = frame_annotation(run, t)
    camera_title = "camera 第一人称视角"
    draw.text((20, 8), camera_title, font=font, fill=(232, 241, 252))
    title_box = draw.textbbox((20, 8), camera_title, font=font)
    dot_x, dot_y = title_box[2] + 24, HEADER_HEIGHT // 2
    color = RECORDING_RED if annotation["recording"] else IDLE_GREEN
    draw.ellipse((dot_x - 12, dot_y - 12, dot_x + 12, dot_y + 12),
                 fill=color, outline=(255, 255, 255), width=2)
    draw.text((dot_x + 21, 10), "拍摄中" if annotation["recording"] else "空闲",
              font=small, fill=color)
    draw.text((width + 20, 8), "环境第三人称视角", font=font, fill=(232, 241, 252))
    draw.line((width, HEADER_HEIGHT, width, height + HEADER_HEIGHT), fill=(19, 26, 35), width=3)
    if side_views is not None:
        if len(side_views) != 2 or any(view.shape != left.shape for view in side_views):
            raise ValueError("the bottom row requires two complete synchronized side views")
        row_y = height + HEADER_HEIGHT
        for column, (view, label) in enumerate(zip(side_views, ("左侧第三人称视角", "右侧第三人称视角"))):
            frame.paste(Image.fromarray(view), (column * width, row_y + HEADER_HEIGHT))
            draw.text((column * width + 20, row_y + 8), label, font=font, fill=(232, 241, 252))
        draw.line((width, row_y + HEADER_HEIGHT, width, footer_y), fill=(19, 26, 35), width=3)
    title, combo = DEMOS[name]
    draw.text((20, footer_y + 7), f"{title} + {combo}  |  物理时间 {t:06.2f} s  |  {annotation['label']}",
              font=small, fill=(210, 223, 238))
    if annotation["recording"]:
        errors = (f"实时轨迹跟踪误差  |  位置 {annotation['position_error_cm']:.2f} cm"
                  f"  |  姿态 {annotation['orientation_error_deg']:.2f}°")
        draw.text((20, footer_y + 39), errors, font=small, fill=(255, 226, 155))
    view_count_label = "四视角" if side_views is not None else "双视角"
    draw.text((width + 20, footer_y + 7), f"成功运行的实测轨迹重放 · 同步{view_count_label} · 1×", font=small,
              fill=(161, 183, 207))
    draw.text((width + 20, footer_y + 39), "红点：运镜拍摄  ·  绿点：空闲 / arc 衔接",
              font=small, fill=(161, 183, 207))
    if path_inset:
        draw_path_inset(draw, run, t, (2 * width - 304, footer_y - 252,
                                     2 * width - 14, footer_y - 12))
    return np.asarray(frame)


def render_one(args, app, name, urdf):
    from PIL import Image
    import omni.kit.app
    import omni.replicator.core as rep
    import jvs_dynamic_simview_isaac as runner
    import jvs_isaac_sim as jvs

    folder = args.runs / name
    run = load_run(folder)
    original_layers = json.loads((folder / "usd_scene_audit.json").read_text())["usd_layer_sha256"]
    if any(digest(path) != expected for path, expected in original_layers.items()):
        raise ValueError("source USD assets differ from the successful audited run")
    stage_file = Path(json.loads((folder / "run.json").read_text())["stage_view"])
    stage, _ = runner.prepare_composed_stage(SimpleNamespace(
        simview_stage=str(stage_file), trajectory_file="", add_ground_plane=False), app)
    robot_path = jvs.import_robot(urdf)
    app.update()
    robot = RobotReplay(stage, robot_path, urdf)
    first_camera = runner.create_optical_camera(stage, robot_path,
        SimpleNamespace(camera_name="DualViewOpticalCamera", camera_vfov=55.))
    third_camera, third_pose = create_environment_camera(stage, run)
    cameras = [first_camera, third_camera]
    side_poses = {}
    if args.layout == "quad":
        side_cameras, side_poses = create_side_cameras(stage, run)
        cameras.extend(side_cameras)
    movers = create_movers(stage, run)
    products, annotators = [], []
    for camera in cameras:
        product = rep.create.render_product(camera, (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        products.append(product)
        annotators.append(annotator)
    robot.update(run, 0.)
    update_movers(run, movers, 0.)
    # Let scene references, shaders, and both RGB products finish loading.
    for _ in range(30):
        app.update()
    rep.orchestrator.step(rt_subframes=8, delta_time=0., pause_timeline=True)
    times = ([0., 14., 28.] if args.preview_only else
             np.arange(int(math.ceil(run.duration * args.fps)) + 1) / args.fps)
    times = np.minimum(times, run.duration)
    output = args.output / f"{name}_{DEMOS[name][1]}_{args.layout}_view.mp4"
    frame_height = (2 if args.layout == "quad" else 1) * (args.height + HEADER_HEIGHT) + FOOTER_HEIGHT
    writer = None
    if not args.preview_only:
        writer = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{2 * args.width}x{frame_height}", "-r", str(args.fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    hashes = {p.name: digest(p) for p in [folder / "telemetry.csv", folder / "obstacle_truth.csv",
        folder / "obstacle_scene.json", folder / "trajectory.json", folder / "reference.json", folder / "run.json"]}
    started = time.monotonic()
    max_camera_error = 0.
    last_progress = started
    count = 0
    annotations = []
    try:
        for index, t in enumerate(times):
            error = robot.update(run, t)
            max_camera_error = max(max_camera_error, error)
            if error > .001:
                raise ValueError(f"URDF replay camera mismatch at t={t}: {error} m")
            update_movers(run, movers, t)
            rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=True, wait_for_render=True)
            images = [np.asarray(annotator.get_data())[..., :3].copy() for annotator in annotators]
            if any(im.shape != (args.height, args.width, 3) for im in images):
                raise RuntimeError("all cameras must return a complete synchronized RGB frame")
            frame = caption_frame(*images[:2], name, run, float(t), path_inset=args.path_inset,
                                  side_views=images[2:] if args.layout == "quad" else None)
            annotations.append(dict(frame=index, physical_time_s=float(t), **frame_annotation(run, float(t))))
            if args.preview_only or index in (0, round(14 * args.fps), round(28 * args.fps), len(times) - 1):
                Image.fromarray(frame).save(args.output / "previews" / f"{name}_{t:06.2f}s.png")
            if writer is not None:
                writer.stdin.write(frame.tobytes())
            count += 1
            if time.monotonic() - last_progress >= 15 or index == len(times) - 1:
                print(json.dumps(dict(demo=name, frame=count, total=len(times), physical_time_s=float(t),
                    elapsed_wall_s=round(time.monotonic() - started, 1))), flush=True)
                last_progress = time.monotonic()
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError(f"video encoding failed: {output}")
        # Kit owns both products until process shutdown. Isaac 5.1's explicit
        # RGB detach can raise an invalid-node error after synchronous steps.
    result = dict(demo=name, combo=DEMOS[name][1], video=output.name if writer is not None else None,
        source_run=str(folder), source_sha256=hashes,
        source_acceptance_mode=('user_authorized_relaxed_acceptance' if args.allow_nonstrict_source
                                else 'strict_independent_audit'),
        strict_source_audit_pass=next(row['dynamic_demo_pass'] for row in args.source_reports
                                      if row['name'] == name),
        renderer_script_sha256=digest(__file__), source_usd_layers_sha256=original_layers,
        rendering="Isaac Sim RTX rendering of successful measured telemetry; not a new closed-loop run",
        left="URDF camera_optical_frame optical camera, 55 degree vertical field of view",
        right="fixed elevated third-person environment camera", environment_camera_pose=third_pose,
        layout=args.layout, view_count=len(cameras),
        texture_streaming_memory_fraction=args.texture_memory_fraction,
        side_camera_poses=side_poses,
        side_camera_reference="left/right of the initial base heading; fixed observers throughout replay",
        measured_path_inset=args.path_inset,
        path_inset_source="measured base and obstacle centers in initial-base XY; trails stop at the frame timestamp",
        synchronization="same paused USD state rendered by all cameras at every physical timestamp",
        interpolation="linear positions/joint angles, SLERP base orientation; measured obstacle truth positions",
        passive_casters="unrecorded passive caster angles remain at URDF neutral pose",
        maximum_camera_fk_interpolation_error_m=max_camera_error,
        recorded_physical_duration_s=run.duration, playback_speed=1, fps=args.fps,
        frames=count, resolution=[2 * args.width, frame_height], stage=str(stage_file),
        annotation_source="last telemetry sample at or before the physical timestamp; no future phase sample",
        shooting_indicator="red only for active orbit/pan/moveout shots; green for arc and idle",
        displayed_errors="measured position error in cm and orientation error in degrees; shooting only",
        arc_source="tracking transition_active samples inside reference.json execution_shots inter-shot windows",
        arc_meaning="inter-shot whole-body reconfiguration; camera may hold its pose while the base repositions",
        frame_annotations=f"{name}_frame_annotations.json",
        phase_frame_counts=dict(Counter(item["phase"] for item in annotations)),
        maximum_annotation_age_s=max((item["physical_time_s"] - item["sample_wall_elapsed_s"]
            for item in annotations if item["sample_wall_elapsed_s"] is not None), default=0.),
        source_files_unchanged=all(digest(folder / name) == value for name, value in hashes.items()),
        source_usd_assets_unchanged=all(digest(path) == expected for path, expected in original_layers.items()))
    (args.output / f"{name}_recording.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / f"{name}_frame_annotations.json").write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", choices=[*DEMOS, "all"], default="all")
    parser.add_argument("--layout", choices=["dual", "quad"], default="dual",
                        help="quad adds fixed left/right third-person views in a second row")
    parser.add_argument("--texture-memory-fraction", type=float,
                        help="optional texture streaming budget as a fraction of GPU memory")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--path-inset", action="store_true",
                        help="overlay measured base/obstacle trails up to each frame timestamp")
    parser.add_argument("--allow-nonstrict-source", action="store_true",
                        help="record a completed run under explicitly authorized relaxed acceptance; preserve its failed checks")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    args.runs = args.runs.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if min(args.fps, args.width, args.height) <= 0 or args.height % 2:
        parser.error("positive frame rate/dimensions and an even height required")
    if args.texture_memory_fraction is not None and not 0 < args.texture_memory_fraction <= 1:
        parser.error("texture memory fraction must be in (0, 1]")
    reports = json.loads((args.runs / "comparison.json").read_text())
    args.source_reports = reports
    names = list(DEMOS) if args.demo == "all" else [args.demo]
    for name in names:
        report = next(row for row in reports if row["name"] == name)
        if not report["dynamic_demo_pass"] and not (args.allow_nonstrict_source
                and report.get('completed') and not report.get('failure')):
            parser.error(f"source demo has not passed its independent audit: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "previews").mkdir(exist_ok=True)
    if args.demo == "all":
        # Separate Kit processes avoid stale render graphs across stage changes.
        recordings = []
        for name in names:
            command = [sys.executable, str(Path(__file__).resolve()), "--runs", str(args.runs),
                "--output", str(args.output), "--demo", name, "--fps", str(args.fps),
                "--width", str(args.width), "--height", str(args.height), "--layout", args.layout]
            if args.preview_only:
                command.append("--preview-only")
            if args.allow_nonstrict_source:
                command.append("--allow-nonstrict-source")
            if args.path_inset:
                command.append("--path-inset")
            if args.texture_memory_fraction is not None:
                command.extend(["--texture-memory-fraction", str(args.texture_memory_fraction)])
            metadata = args.output / f"{name}_recording.json"
            child_started = time.time_ns()
            subprocess.run(command, check=True)
            if not metadata.is_file() or metadata.stat().st_mtime_ns < child_started:
                raise RuntimeError(f"renderer did not finish saving {name}")
            result = json.loads(metadata.read_text())
            if not args.preview_only and not result.get("video"):
                raise RuntimeError(f"renderer did not finish recording {name}")
            recordings.append(result)
        (args.output / "recordings.json").write_text(json.dumps(recordings, ensure_ascii=False, indent=2) + "\n")
        return
    sys.argv[:] = [sys.argv[0]]
    from isaacsim import SimulationApp

    app = SimulationApp(dict(headless=True, width=args.width, height=args.height,
        renderer="RayTracedLighting", anti_aliasing=3, hide_ui=True, disable_viewport_updates=True))
    try:
        if args.texture_memory_fraction is not None:
            import carb.settings
            settings = carb.settings.get_settings()
            settings.set_bool("/rtx-transient/resourcemanager/enableTextureStreaming", True)
            settings.set_float("/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
                               args.texture_memory_fraction)
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.asset.importer.urdf", True)
        import jvs_isaac_sim as jvs
        with tempfile.TemporaryDirectory(prefix="jvs_dual_view_") as temporary:
            urdf = Path(temporary) / "robot.urdf"
            jvs.expand_xacro(SimpleNamespace(xacro=str(PACKAGE / "urdf/jvs_mock.urdf.xacro"),
                xacro_executable="/opt/ros/humble/bin/xacro"), urdf)
            recordings = [render_one(args, app, name, urdf) for name in names]
            manifest = args.output / "recordings.json"
            existing = json.loads(manifest.read_text()) if manifest.is_file() else []
            combined = {item['demo']: item for item in existing}
            combined.update({item['demo']: item for item in recordings})
            manifest.write_text(json.dumps([combined[name] for name in DEMOS if name in combined],
                ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
