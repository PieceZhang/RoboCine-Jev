#!/usr/bin/env python3
"""Run the JVS articulation inside a RoboCine or source USD stage.

The regular JVS Isaac runner owns a native Z-up stage, while
``tools/simview_isaac.py`` authors Y-up stages.  This runner keeps PhysX and
the imported URDF in their native Z-up frame.  It mounts a Y-up source below
a +90 degree X rotation and mounts an already-Z-up source unchanged.  The
source's declared default prim is used instead of assuming it is ``/World``.
It also creates a real USD camera below the URDF's otherwise-empty
``camera_optical_frame``.

The ROS command graph is the same as ``jvs_isaac_sim.py``.  Consequently the
existing ``jvs_follow_traj`` node can command the six-axis camera arm without
special Isaac-side trajectory code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


def boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def focal_from_vfov(vfov_deg: float) -> float:
    """Vertical FOV to focal length for simview's 36 x 20.25 mm aperture."""
    if not math.isfinite(vfov_deg) or not 1.0 < vfov_deg < 179.0:
        raise ValueError("camera-vfov must be between 1 and 179 degrees")
    return (20.25 * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)


def simview_to_isaac_spawn(
    x: float, z: float, yaw: float, height: float = 0.002
) -> tuple[float, float, float, float]:
    """Convert a Y-up RoboCine floor pose to a Z-up Isaac/URDF floor pose.

    RoboCine base heading ``yaw=0`` points toward world +Z.  After the
    ``Rx(+90deg)`` stage conversion that direction is Isaac -Y, hence the
    additional -pi/2 yaw offset for the URDF's +X-forward base.
    """
    values = (x, z, yaw, height)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("spawn pose values must be finite")
    return float(x), float(-z), float(height), float(yaw - math.pi * 0.5)


def build_parser() -> argparse.ArgumentParser:
    # The sibling module imports only the standard library before Kit starts.
    import jvs_isaac_sim as jvs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xacro", required=True, help="JVS robot Xacro file")
    parser.add_argument(
        "--xacro-executable", default="xacro", help="xacro executable or absolute path"
    )
    parser.add_argument(
        "--simview-stage",
        required=True,
        help="Y-up simview stage or Z-up source USD with a default/root prim",
    )
    parser.add_argument("--save-stage", default="", help="optional composed USD output")
    parser.add_argument("--headless", type=boolean, default=False)
    parser.add_argument("--add-ground-plane", type=boolean, default=True)
    parser.add_argument(
        "--ground-visible",
        type=boolean,
        default=False,
        help="show the PhysX ground plane (collision remains enabled when hidden)",
    )
    parser.add_argument("--physics-hz", type=float, default=1000.0)
    jvs.add_physics_arguments(parser, solver_type='PGS')
    parser.add_argument('--camera-motion-audit-file', default='',
                        help='optional unfiltered camera/robot telemetry at every physics step')
    parser.add_argument('--real-time-factor', type=float, default=.1,
                        help='maximum simulation/wall-time rate; leaves ROS scheduling headroom')
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="stop after N app updates; 0 runs until Isaac Sim exits",
    )
    parser.add_argument("--spawn-x", type=float, default=5.0,
                        help="initial base X in the RoboCine Y-up world")
    parser.add_argument("--spawn-z", type=float, default=1.0,
                        help="initial base Z in the RoboCine Y-up world")
    parser.add_argument("--spawn-yaw", type=float, default=0.0,
                        help="initial base yaw in RoboCine radians (zero points +Z)")
    parser.add_argument("--spawn-height", type=float, default=0.002,
                        help="URDF base clearance above the floor in metres")
    parser.add_argument("--camera-vfov", type=float, default=55.0)
    parser.add_argument("--camera-name", default="RoboCineCamera")
    parser.add_argument("--activate-camera", type=boolean, default=True,
                        help="select the optical camera in the GUI viewport")
    parser.add_argument(
        "--viewport-scroll-scale",
        type=float,
        default=0.2,
        help="ordinary mouse-wheel dolly/zoom sensitivity; Isaac default is 1",
    )
    parser.add_argument(
        "--viewport-drag-scale",
        type=float,
        default=1.0,
        help=(
            "mouse-drag sensitivity for pan, orbit, look, and dolly; "
            "Isaac default is 1"
        ),
    )
    parser.add_argument(
        "--viewport-speed-step",
        type=float,
        default=1.02,
        help="RMB+wheel camera-speed multiplier; Isaac default is 1.1",
    )
    parser.add_argument(
        "--trajectory-file", default="",
        help="optional jvs_follow_traj JSON; used to keep the timeline long enough",
    )
    parser.add_argument("--trajectory-start-delay", type=float, default=1.0)
    parser.add_argument("--reset-service", default="/jvs/reset_robot")
    parser.add_argument("--obstacle-trajectory-file", default="",
                        help="optional jvs_dynamic_scene/v1 obstacle models and trajectories")
    parser.add_argument("--dynamic-audit-file", default="",
                        help="optional physical-time obstacle truth CSV for independent evaluation")
    parser.add_argument("--enable-lidar", type=boolean, default=True)
    parser.add_argument("--lidar-config", default=jvs.DEFAULT_LIDAR_CONFIG)
    parser.add_argument("--lidar-variant", default="")
    parser.add_argument("--lidar-topic", default="/lidar/points")
    parser.add_argument("--lidar-frame-id", default="lidar_link")
    parser.add_argument("--lidar-full-scan", type=boolean, default=True)
    parser.add_argument("--lidar-motion-bvh", type=boolean, default=False)
    parser.add_argument("--lidar-debug-view", type=boolean, default=False)
    jvs.add_front_lidar_arguments(parser)
    parser.add_argument(
        "--record-dir", default="",
        help="record the optical camera first-person view to this directory "
        "(PNG frames + fp_cam.mp4); empty disables recording",
    )
    parser.add_argument(
        "--record-fps", type=float, default=30.0,
        help="wall-clock capture rate for the first-person recording",
    )
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    from jvs_isaac_sim import validate_front_lidar_arguments, validate_physics_arguments
    validate_front_lidar_arguments(args)
    validate_physics_arguments(args)
    xacro = Path(args.xacro).expanduser().resolve()
    if not xacro.is_file():
        raise FileNotFoundError(f"Xacro file does not exist: {xacro}")
    args.xacro = str(xacro)
    stage = Path(args.simview_stage).expanduser().resolve()
    if not stage.is_file():
        raise FileNotFoundError(f"simview stage does not exist: {stage}")
    args.simview_stage = str(stage)
    if os.path.sep in args.xacro_executable:
        executable = Path(args.xacro_executable).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"xacro executable does not exist: {executable}")
        args.xacro_executable = str(executable)
    elif shutil.which(args.xacro_executable) is None:
        raise FileNotFoundError(f"cannot find xacro executable: {args.xacro_executable}")
    if args.physics_hz <= 0.0:
        raise ValueError("physics-hz must be positive")
    if not math.isfinite(args.real_time_factor) or not 0 < args.real_time_factor <= 1.:
        raise ValueError('real-time-factor must be in (0,1]')
    if args.max_steps < 0:
        raise ValueError("max-steps must not be negative")
    if not args.camera_name or "/" in args.camera_name:
        raise ValueError("camera-name must be one valid USD prim name")
    if args.trajectory_start_delay < 0.0:
        raise ValueError("trajectory-start-delay must not be negative")
    if not args.reset_service.strip():
        raise ValueError("reset-service must not be empty")
    if (
        not math.isfinite(args.viewport_scroll_scale)
        or args.viewport_scroll_scale <= 0.0
    ):
        raise ValueError("viewport-scroll-scale must be finite and positive")
    if (
        not math.isfinite(args.viewport_drag_scale)
        or args.viewport_drag_scale <= 0.0
    ):
        raise ValueError("viewport-drag-scale must be finite and positive")
    if (
        not math.isfinite(args.viewport_speed_step)
        or args.viewport_speed_step <= 1.0
    ):
        raise ValueError(
            "viewport-speed-step must be finite and greater than 1"
        )
    if args.trajectory_file:
        trajectory = Path(args.trajectory_file).expanduser().resolve()
        if not trajectory.is_file():
            raise FileNotFoundError(f"trajectory file does not exist: {trajectory}")
        args.trajectory_file = str(trajectory)
    if args.record_dir:
        args.record_dir = str(Path(args.record_dir).expanduser().resolve())
        if args.record_fps <= 0.0:
            raise ValueError("record-fps must be positive")
        if args.record_width <= 0 or args.record_height <= 0:
            raise ValueError("record-width/record-height must be positive")
    focal_from_vfov(args.camera_vfov)
    simview_to_isaac_spawn(
        args.spawn_x, args.spawn_z, args.spawn_yaw, args.spawn_height
    )


def source_reference_prim(source) -> str:
    """Return the source prim that should be mounted in the Isaac stage."""
    default_prim = source.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return str(default_prim.GetPath())

    world = source.GetPrimAtPath("/World")
    if world and world.IsValid():
        return "/World"

    root_prims = [
        prim
        for prim in source.GetPseudoRoot().GetChildren()
        if prim.IsActive()
    ]
    if len(root_prims) == 1:
        return str(root_prims[0].GetPath())

    paths = [str(prim.GetPath()) for prim in root_prims]
    raise RuntimeError(
        "source USD must declare a defaultPrim, contain /World, or have "
        f"exactly one active root prim; found {paths}"
    )


def source_to_isaac_rotation(up_axis: str) -> float:
    """Return the X rotation that converts a source stage to Isaac Z-up."""
    normalized = str(up_axis).strip().upper()
    if normalized == "Y":
        return 90.0
    if normalized == "Z":
        return 0.0
    raise ValueError(f"source USD upAxis must be Y or Z, got {up_axis!r}")


def prepare_composed_stage(args: argparse.Namespace, simulation_app):
    import omni.usd
    from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdLux

    context = omni.usd.get_context()
    context.new_stage()
    simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not create a USD stage")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    source = Usd.Stage.Open(args.simview_stage, load=Usd.Stage.LoadNone)
    if source is None:
        raise RuntimeError(f"could not open simview stage: {args.simview_stage}")
    source_prim_path = source_reference_prim(source)
    source_up_axis = UsdGeom.GetStageUpAxis(source)
    source_rotation = source_to_isaac_rotation(source_up_axis)
    stage.SetStartTimeCode(source.GetStartTimeCode())
    stage.SetEndTimeCode(source.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(source.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(source.GetFramesPerSecond())
    if args.trajectory_file:
        trajectory = json.loads(
            Path(args.trajectory_file).read_text(encoding="utf-8")
        )
        duration = float(trajectory.get("duration_s", 0.0))
        if duration <= 0.0:
            raise ValueError("trajectory JSON must contain a positive duration_s")
        # The timeline drives /clock.  Several simview shots end at 6 s, the
        # same duration as the default arm path, so the follower's startup
        # delay would otherwise make Kit stop one second before completion.
        needed_seconds = args.trajectory_start_delay + duration + 1.0
        needed_end = source.GetStartTimeCode() + (
            needed_seconds * source.GetTimeCodesPerSecond()
        )
        stage.SetEndTimeCode(max(stage.GetEndTimeCode(), needed_end))

    # Reference, rather than copy, the complete source stage.  A Y-up simview
    # needs Rx(+90deg), mapping (x, y_up, z) to (x, -z, y_up).  Native Z-up
    # stages (such as InteriorAgent) must not be rotated.
    simview = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Simview"))
    simview.GetPrim().GetReferences().AddReference(
        args.simview_stage, source_prim_path
    )
    if source_rotation:
        simview.AddRotateXOp().Set(source_rotation)
    # Exported shot previews are drawing aids, not scene obstacles. RTX rays
    # otherwise hit the animated frustum lines, including inside the real JVS.
    for name in ('CamBody', 'Annotations'):
        helper = stage.GetPrimAtPath('/World/Simview/' + name)
        if helper:
            UsdGeom.Imageable(helper).MakeInvisible()
    print(
        "JVS simview: mounted source "
        f"prim={source_prim_path}, up_axis={source_up_axis}, "
        f"rotate_x={source_rotation:g}",
        flush=True,
    )

    if args.add_ground_plane:
        PhysicsSchemaTools.addGroundPlane(
            stage,
            "/World/groundPlane",
            "Z",
            50.0,
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.35, 0.35, 0.35),
        )
        if not args.ground_visible:
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/groundPlane")).MakeInvisible()
    if not stage.GetPrimAtPath("/World/DistantLight"):
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        light.CreateIntensityAttr(500.0)
    return stage, source_rotation


def find_robot_prim(stage, robot_path: str, name: str):
    """Find one named URDF prim inside the imported robot namespace."""
    from pxr import Sdf

    robot_scope = Sdf.Path(robot_path).GetParentPath()
    matches = [
        prim for prim in stage.Traverse()
        if prim.GetName() == name and prim.GetPath().HasPrefix(robot_scope)
    ]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(
            f"expected one {name!r} below {robot_scope}, found {len(matches)}: {paths}"
        )
    return matches[0]


def create_optical_camera(stage, robot_path: str, args: argparse.Namespace) -> str:
    """Create an RTX/viewport camera below the URDF optical-frame link."""
    from pxr import Gf, UsdGeom

    optical = find_robot_prim(stage, robot_path, "camera_optical_frame")
    camera_path = optical.GetPath().AppendChild(args.camera_name)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateFocalLengthAttr(focal_from_vfov(args.camera_vfov))
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)

    # URDF optical frame: +Z forward, +X image-right, +Y image-down.
    # USD camera: -Z forward, +X image-right, +Y image-up.
    UsdGeom.Xformable(camera.GetPrim()).AddRotateXOp().Set(180.0)
    camera.GetPrim().SetCustomDataByKey("robocine:rosFrameId", "camera_optical_frame")
    return str(camera_path)


def activate_viewport_camera(camera_path: str, simulation_app) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        for _ in range(5):
            simulation_app.update()
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("no active viewport")
        viewport.set_active_camera(camera_path)
    except Exception as error:
        print(f"JVS simview: could not activate viewport camera: {error}", flush=True)


def create_fp_recorder(camera_path: str, args: argparse.Namespace):
    """Attach a Replicator RGB annotator + live mp4 writer to the optical camera.

    Frames go to fp_cam.mp4 in real time AND to PNGs (re-mux fallback if a
    signal kills the process before the writer is released — no ffmpeg on
    this host, cv2 keeps the mp4 headerless until release).
    """
    import cv2
    import omni.replicator.core as rep

    render_product = rep.create.render_product(
        camera_path, (args.record_width, args.record_height)
    )
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    Path(args.record_dir).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(Path(args.record_dir) / "fp_cam.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.record_fps,
        (args.record_width, args.record_height),
    )
    print(
        f"JVS simview: first-person recording → {args.record_dir} "
        f"({args.record_width}x{args.record_height} @ {args.record_fps:g}fps)",
        flush=True,
    )
    return annotator, writer, render_product


def capture_fp_frame(recorder, args: argparse.Namespace, index: int) -> bool:
    """Persist one annotator frame (mp4 + PNG); False while the buffer is empty."""
    import cv2
    import numpy as np
    from PIL import Image

    annotator, writer, _product = recorder
    data = annotator.get_data()
    if data is None or data.size == 0:
        return False
    frame = np.asarray(data)[..., :3].copy()  # drop alpha; copy: reused buffer
    if frame.shape[0] != args.record_height or frame.shape[1] != args.record_width:
        return False
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    Image.fromarray(frame).save(
        Path(args.record_dir) / f"fp_cam_{index:05d}.png"
    )
    return True


def encode_fp_recording(recorder, args: argparse.Namespace, n_frames: int) -> None:
    """Release the live mp4 writer; PNGs remain as the re-mux fallback."""
    _annotator, writer, _product = recorder
    writer.release()
    print(
        f"JVS simview: first-person video saved: "
        f"{Path(args.record_dir) / 'fp_cam.mp4'} ({n_frames} frames)",
        flush=True,
    )






def configure_viewport_navigation(
    scroll_scale: float, drag_scale: float, speed_step: float
) -> None:
    """Reduce mouse navigation deltas for metre-scale scenes."""
    import carb
    from omni.kit.manipulator.camera import gestures
    from omni.kit.manipulator.camera import viewport_camera_manipulator

    settings = carb.settings.get_settings()
    settings.set(
        "/persistent/app/viewport/camVelocityScalerMultAmount", speed_step
    )

    # Isaac 5.1 exposes no setting for mouse-drag sensitivity.  Scale the
    # gesture deltas process-locally; this covers pan, orbit/tumble, RMB look,
    # and Alt+RMB dolly regardless of the active viewport bindings.
    gesture_names = ("PanGesture", "TumbleGesture", "LookGesture", "ZoomGesture")
    for class_name in gesture_names:
        gesture_class = getattr(gestures, class_name)
        original_name = "_jvs_unscaled_on_mouse_move"
        original = getattr(
            gesture_class, original_name, gesture_class.on_mouse_move
        )
        setattr(gesture_class, original_name, original)

        def scaled_mouse_move(self, mouse_moved, _original=original):
            return _original(
                self,
                (mouse_moved[0] * drag_scale, mouse_moved[1] * drag_scale),
            )

        gesture_class.on_mouse_move = scaled_mouse_move

    # ViewportEventDelegate imports this function for every ordinary wheel
    # event.  Wheel zoom internally uses ZoomGesture, which is already scaled
    # above, so compensate here to retain an independently tunable final scale.
    original_name = "_jvs_unscaled_zoom_operation"
    original = getattr(
        viewport_camera_manipulator,
        original_name,
        viewport_camera_manipulator._zoom_operation,
    )
    setattr(viewport_camera_manipulator, original_name, original)

    def scaled_zoom_operation(x, y, viewport_api):
        return original(x, y * scroll_scale / drag_scale, viewport_api)

    viewport_camera_manipulator._zoom_operation = scaled_zoom_operation
    print(
        "JVS simview: viewport navigation tuning: "
        f"scroll_scale={scroll_scale:g}, drag_scale={drag_scale:g}, "
        f"speed_step={speed_step:g}",
        flush=True,
    )


def create_dynamic_lidar(stage, robot_path, config, link, args, jvs):
    if config not in ('JVS_DynamicWide', 'JVS_DynamicNearfield'):
        return (jvs.create_front_rtx_lidar(stage, robot_path, args) if link == 'front_lidar_link'
                else jvs.create_rtx_lidar(stage, robot_path, args))
    import carb.settings
    import omni.kit.commands
    from pxr import Sdf
    folder = str(Path(__file__).resolve().parents[1] / 'config')
    settings = carb.settings.get_settings()
    folders = list(settings.get(jvs.LIDAR_PROFILE_SETTING) or [])
    settings.set_string_array(jvs.LIDAR_PROFILE_SETTING, [folder, *[p for p in folders if p != folder]])
    parent = Sdf.Path(robot_path).GetParentPath().AppendChild(link)
    ok, sensor = omni.kit.commands.execute('IsaacSensorCreateRtxLidar',
        path='/JvsRtxLidar', parent=str(parent), config=config,
        visibility=False, force_camera_prim=True)
    if not ok or sensor is None or not sensor.IsValid():
        raise RuntimeError('failed to create dynamic LiDAR ' + config)
    return str(sensor.GetPath())


def run(args: argparse.Namespace) -> int:
    # The bridge selects its ROS ABI while Kit starts, so these must exist first.
    os.environ.setdefault("ROS_DISTRO", "humble")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "enable_motion_bvh": args.enable_lidar and args.lidar_motion_bvh,
        }
    )
    reset_server = None
    camera_audit = None
    dynamic_obstacles = None
    static_audit = None
    from isaac_dynamic_scene import prepare_static_scene, finish_static_scene_audit
    try:
        # This sibling module has no Omniverse imports at module load time and
        # remains the single implementation of URDF/PhysX/ROS graph setup.
        import jvs_isaac_sim as jvs

        print("JVS simview: enabling Isaac extensions", flush=True)
        jvs.enable_extensions(simulation_app)
        if not args.headless:
            configure_viewport_navigation(
                args.viewport_scroll_scale,
                args.viewport_drag_scale,
                args.viewport_speed_step,
            )
        with tempfile.TemporaryDirectory(prefix="jvs_simview_isaac_") as folder:
            urdf_path = Path(folder) / "jvs_mock.urdf"
            jvs.expand_xacro(args, urdf_path)
            stage, source_rotation = prepare_composed_stage(
                args, simulation_app
            )
            print(f"JVS simview: referenced {args.simview_stage}", flush=True)
            static_audit = prepare_static_scene(stage, args.dynamic_audit_file,
                (args.spawn_x, -args.spawn_z, args.spawn_height))
            robot_path = jvs.import_robot(urdf_path)
            simulation_app.update()

            scene_spawn = {
                "x_m": float(args.spawn_x),
                "z_m": float(args.spawn_z),
                "yaw_rad": float(args.spawn_yaw),
                "height_m": float(args.spawn_height),
                "arm_joint_positions_rad": [0.0] * 6,
            }
            # configure_articulation expects native Isaac spawn coordinates.
            args.spawn_x, args.spawn_y, args.spawn_z, args.spawn_yaw = (
                simview_to_isaac_spawn(
                    args.spawn_x, args.spawn_z, args.spawn_yaw, args.spawn_height
                )
            )
            jvs.configure_articulation(stage, robot_path, args, urdf_path)
            jvs.configure_physics_rate(stage, args.physics_hz, args.solver_type)
            if args.obstacle_trajectory_file:
                from isaac_dynamic_obstacles import IsaacDynamicObstacles
                dynamic_obstacles = IsaacDynamicObstacles(stage, args.obstacle_trajectory_file,
                    dict(x=scene_spawn['x_m'], z=scene_spawn['z_m'],
                         yaw=scene_spawn['yaw_rad'], height=scene_spawn['height_m']), args.dynamic_audit_file)
            lidar_path = ""
            front_lidar_path = ""
            if args.enable_lidar:
                if "rear" in jvs.enabled_lidars(args):
                    lidar_path = create_dynamic_lidar(stage, robot_path, args.lidar_config, 'lidar_link', args, jvs)
                if 'front' in jvs.enabled_lidars(args):
                    front_lidar_path = create_dynamic_lidar(stage, robot_path, args.front_lidar_config, 'front_lidar_link', args, jvs)
                simulation_app.update()
                print(
                    "JVS simview: RTX LiDAR created at "
                    f"rear={lidar_path or 'disabled'} (config={args.lidar_config}); "
                    f"front={front_lidar_path or 'disabled'} (config={args.front_lidar_config})",
                    flush=True,
                )
            jvs.create_ros_graph(
                robot_path,
                lidar_path=lidar_path,
                lidar_topic=args.lidar_topic,
                lidar_frame_id=args.lidar_frame_id,
                lidar_full_scan=args.lidar_full_scan,
                lidar_debug_view=args.lidar_debug_view,
                front_lidar_path=front_lidar_path,
                front_lidar_topic=args.front_lidar_topic,
                filtered_commands=args.command_filter_hz > 0 or args.physics_feedback_hz > 0,
                physics_feedback=args.physics_feedback_hz > 0,
            )
            camera_path = create_optical_camera(stage, robot_path, args)
            print(
                "JVS simview: optical camera created at " + camera_path,
                flush=True,
            )
            # A report camera shows the complete robot and moving obstacles.
            # Its pose follows the declared initial frame, never the live base.
            from pxr import Gf, UsdGeom
            report_camera = UsdGeom.Camera.Define(stage, '/World/DynamicReportCamera')
            report_camera.AddTranslateOp().Set(Gf.Vec3d(args.spawn_x, args.spawn_y, 2.4))
            report_camera.CreateFocalLengthAttr(23.)
            report_camera.CreateHorizontalApertureAttr(42.)
            report_camera.CreateVerticalApertureAttr(31.5)

            if args.save_stage:
                output = Path(args.save_stage).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                stage.GetRootLayer().Export(str(output))
                print(f"JVS simview: composed stage saved to {output}", flush=True)

            if not args.headless and args.activate_camera:
                activate_viewport_camera(camera_path, simulation_app)

            import omni.timeline
            from isaacsim.core.simulation_manager import SimulationManager

            timeline = omni.timeline.get_timeline_interface()
            timeline.set_play_every_frame(True)
            timeline.set_target_framerate(60.)
            timeline.play()
            # Warm up PhysX/tensor handles without an app update (which would
            # render startup transients and tick the ROS command graph).
            # Restore and sleep the robot before its first visible frame.
            SimulationManager.initialize_physics()
            reset_server = jvs.RobotResetService(
                robot_path,
                (args.spawn_x, args.spawn_y, args.spawn_z, args.spawn_yaw),
                scene_spawn,
                args.reset_service,
            )
            reset_server.hold_startup_pose()
            if dynamic_obstacles is not None:
                dynamic_obstacles.attach(reset_server.node)
            if args.physics_feedback_hz > 0:
                jvs.PhysicsFeedback(reset_server, args.physics_feedback_hz,
                                    initial_time=dynamic_obstacles.time if dynamic_obstacles else 0.)
            if args.command_filter_hz > 0 or args.physics_feedback_hz > 0:
                jvs.PhysicsCommandFilter(reset_server, urdf_path, args.command_filter_hz,
                    initial_time=dynamic_obstacles.time if dynamic_obstacles else 0.,
                    audit_file=str(Path(args.camera_motion_audit_file).with_name('command_filter.json'))
                        if args.camera_motion_audit_file else '')
            if args.camera_motion_audit_file:
                camera_audit = jvs.CameraMotionAudit(
                    args.camera_motion_audit_file, reset_server.articulation,
                    str(find_robot_prim(stage, robot_path, 'camera_link').GetPath()),
                    initial_time=dynamic_obstacles.time if dynamic_obstacles else 0.,
                    scene=dynamic_obstacles)
            print('JVS physics settings: ' + json.dumps(dict(
                solver_type=args.solver_type, physics_hz=args.physics_hz,
                solver_velocity_iterations=args.solver_velocity_iterations,
                arm_drive_stiffness=args.arm_drive_stiffness,
                arm_drive_damping=args.arm_drive_damping,
                arm_drive_type=args.arm_drive_type,
                command_filter_hz=args.command_filter_hz,
                physics_feedback_hz=args.physics_feedback_hz,
                wheel_drive_damping=args.wheel_drive_damping,
                actuator_armature=args.actuator_armature)), flush=True)
            print(
                "JVS simview ready: "
                f"articulation={robot_path}, camera={camera_path}, "
                "command_topic=/isaac_joint_commands, "
                f"reset_service={args.reset_service}, "
                f"lidar_topic={args.lidar_topic if lidar_path else 'disabled'}, "
                f"front_lidar_topic={args.front_lidar_topic if front_lidar_path else 'disabled'}",
                flush=True,
            )
            steps = 0
            fp_recorder = (
                create_fp_recorder(str(report_camera.GetPath()), args)
                if args.record_dir else None
            )
            fp_index = 0
            fp_next = 0.0
            frame_times = []
            # Let the Python loop finish the audit and video before Kit closes.
            # Kit's default SIGINT exit can bypass Python finally blocks.
            import signal
            stopping = False
            def request_stop(signum, frame):
                nonlocal stopping
                stopping = True
            signal.signal(signal.SIGINT, request_stop)
            signal.signal(signal.SIGTERM, request_stop)
            try:
                while not stopping and simulation_app.is_running() and (
                    args.max_steps == 0 or steps < args.max_steps
                ):
                    iteration_start = time.monotonic()
                    physical_before = dynamic_obstacles.time if dynamic_obstacles else timeline.get_current_time()
                    capture_requested = False
                    if fp_recorder is not None:
                        active = dynamic_obstacles is None or (dynamic_obstacles.epoch is not None and physical_before >= dynamic_obstacles.epoch)
                        capture_requested = active and physical_before >= fp_next - 1e-9
                    simulation_app.update()
                    reset_server.spin_once()
                    steps += 1
                    if fp_recorder is not None:
                        now = dynamic_obstacles.time if dynamic_obstacles else timeline.get_current_time()
                        if capture_requested:
                            if capture_fp_frame(fp_recorder, args, fp_index):
                                frame_times.append(dict(frame=fp_index, simulation_time_s=now,
                                    scene_time_s=now - dynamic_obstacles.epoch if dynamic_obstacles else now))
                                fp_index += 1
                                fp_next = physical_before + 1.0 / args.record_fps
                            else:
                                fp_next = physical_before  # keep rendering until host RGB is ready
                    physical_after = dynamic_obstacles.time if dynamic_obstacles else timeline.get_current_time()
                    remaining = ((physical_after - physical_before) / args.real_time_factor
                                 - (time.monotonic() - iteration_start))
                    if remaining > 0:
                        time.sleep(remaining)
                timeline.stop()
            finally:
                # runs even on Ctrl-C: releases the mp4 header before close()
                if fp_recorder is not None:
                    encode_fp_recording(fp_recorder, args, fp_index)
                    (Path(args.record_dir) / 'frame_times.json').write_text(json.dumps(frame_times) + '\n')
                if camera_audit is not None:
                    camera_audit.close()
        return 0
    except Exception as error:
        print(
            f"JVS simview failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        finish_static_scene_audit(static_audit, args.dynamic_audit_file)
        if dynamic_obstacles is not None:
            dynamic_obstacles.close()
        if reset_server is not None:
            reset_server.close()
        simulation_app.close()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    # Prevent SimulationApp from interpreting runner-only options.
    sys.argv[:] = [sys.argv[0]]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
