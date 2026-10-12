from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "jvs_mock"


def _load_motion_module():
    path = PKG / "jvs_mock" / "motion.py"
    spec = importlib.util.spec_from_file_location("jvs_mock_motion", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_kinematics_module():
    path = PKG / "jvs_mock" / "jvs_kinematics.py"
    spec = importlib.util.spec_from_file_location("jvs_mock_kinematics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_isaac_bridge_module():
    path = PKG / "jvs_mock" / "isaac_bridge.py"
    spec = importlib.util.spec_from_file_location("jvs_mock_isaac_bridge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_simview_runner_module():
    path = PKG / "scripts" / "jvs_simview_isaac.py"
    spec = importlib.util.spec_from_file_location("jvs_simview_isaac", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_isaac_runner_module():
    path = PKG / "scripts" / "jvs_isaac_sim.py"
    spec = importlib.util.spec_from_file_location("jvs_isaac_sim", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_description_xml_files_are_well_formed():
    for path in [*sorted((PKG / "urdf").glob("*.xacro")), PKG / "worlds" / "empty.world"]:
        ET.parse(path)


def test_simview_to_isaac_spawn_is_a_proper_y_up_to_z_up_mapping():
    runner = _load_simview_runner_module()
    x, y, z, yaw = runner.simview_to_isaac_spawn(1.25, -2.0, 0.0)
    assert (x, y, z) == (1.25, 2.0, 0.002)
    assert yaw == -math.pi / 2.0

    # RoboCine yaw +90 degrees points +X, which is also URDF/Isaac yaw zero.
    _, _, _, yaw = runner.simview_to_isaac_spawn(0.0, 0.0, math.pi / 2.0)
    assert yaw == 0.0


def test_simview_runner_defaults_to_open_living_room_center():
    runner = _load_simview_runner_module()
    parser = runner.build_parser()
    assert parser.get_default("spawn_x") == 5.0
    assert parser.get_default("spawn_z") == 1.0

    # Native Isaac coordinates after the RoboCine Y-up to Z-up conversion.
    x, y, z, yaw = runner.simview_to_isaac_spawn(
        parser.get_default("spawn_x"),
        parser.get_default("spawn_z"),
        parser.get_default("spawn_yaw"),
        parser.get_default("spawn_height"),
    )
    assert (x, y, z) == (5.0, -1.0, 0.002)
    assert yaw == -math.pi / 2.0


@pytest.mark.parametrize("sleep_accepted", [True, False])
def test_isaac_startup_restores_complete_state_before_sleep(monkeypatch, sleep_accepted):
    runner = _load_isaac_runner_module()
    calls = []
    physics = SimpleNamespace(
        put_to_sleep=lambda *args: calls.append(("sleep", args)),
        is_sleeping=lambda *args: sleep_accepted,
    )
    usd = SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage_id=lambda: 17))
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(usd=usd))
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace(
        get_physx_simulation_interface=lambda: physics))
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        PhysicsSchemaTools=SimpleNamespace(sdfPathToInt=lambda path: 23),
        Sdf=SimpleNamespace(Path=lambda path: path),
    ))
    server = runner.RobotResetService.__new__(runner.RobotResetService)
    server._np = np
    server.robot_path = "/robot/base_link"
    server.position = np.array([5.0, -1.0, 0.002])
    server.orientation = np.array([math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)])
    server.joint_names = list(runner.ARM_JOINTS + runner.WHEEL_JOINTS + runner.CASTER_JOINTS)
    server.joint_positions = np.zeros(len(server.joint_names))
    server.articulation = SimpleNamespace(**{
        name: (lambda *args, name=name: calls.append((name, args)))
        for name in ("set_world_pose", "set_linear_velocity", "set_angular_velocity",
                     "set_joint_positions", "set_joint_velocities")
    })
    server.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=lambda msg: None))
    if sleep_accepted:
        server.hold_startup_pose()
    else:
        with pytest.raises(RuntimeError, match="idle state"):
            server.hold_startup_pose()
    assert [name for name, _ in calls] == [
        "set_world_pose", "set_linear_velocity", "set_angular_velocity",
        "set_joint_positions", "set_joint_velocities", "sleep",
    ]
    np.testing.assert_array_equal(calls[0][1][0], server.position)
    np.testing.assert_array_equal(calls[0][1][1], server.orientation)
    for _, args in calls[1:5]:
        assert np.count_nonzero(args[0]) == 0
    assert len(calls[4][1][0]) == len(server.joint_names)
    assert calls[-1][1] == (17, 23)


def test_isaac_runners_enable_the_same_rtx_lidar_contract_by_default():
    isaac = _load_isaac_runner_module()
    simview = _load_simview_runner_module()
    for runner in (isaac, simview):
        parser = runner.build_parser()
        assert parser.get_default("enable_lidar") is True
        assert parser.get_default("lidar_config") == "Example_Rotary"
        assert parser.get_default("lidar_topic") == "/lidar/points"
        assert parser.get_default("lidar_frame_id") == "lidar_link"
        assert parser.get_default("lidar_full_scan") is True
        assert parser.get_default("lidar_motion_bvh") is False
        assert parser.get_default("lidar_debug_view") is False

    source = (PKG / "scripts" / "jvs_isaac_sim.py").read_text()
    for contract in (
        "isaacsim.sensors.rtx",
        '"enable_motion_bvh": args.enable_lidar and args.lidar_motion_bvh',
        "IsaacSensorCreateRtxLidar",
        "isaacsim.core.nodes.IsaacCreateRenderProduct",
        "isaacsim.ros2.bridge.ROS2RtxLidarHelper",
        '"CreateLidarRenderProduct.inputs:width", 128',
        '"CreateLidarRenderProduct.inputs:height", 128',
        '"PublishLidarPointCloud.inputs:type", "point_cloud"',
        '"PublishLidarPointCloud.inputs:useSystemTime", False',
    ):
        assert contract in source

    for launch_name in ("jvs_mock_isaac.launch.py", "jvs_mock_simview.launch.py"):
        launch_source = (PKG / "launch" / launch_name).read_text()
        for argument in (
            "enable_lidar",
            "lidar_config",
            "lidar_topic",
            "lidar_frame_id",
            "lidar_full_scan",
            "lidar_motion_bvh",
            "lidar_debug_view",
        ):
            assert f'DeclareLaunchArgument("{argument}"' in launch_source


def test_simview_optical_camera_focal_matches_vertical_fov():
    runner = _load_simview_runner_module()
    focal = runner.focal_from_vfov(55.0)
    recovered = math.degrees(2.0 * math.atan((20.25 * 0.5) / focal))
    assert math.isclose(recovered, 55.0, abs_tol=1.0e-12)


def test_required_joint_contract_is_present():
    root = ET.parse(PKG / "urdf" / "jvs_mock.urdf.xacro").getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    expected = {
        "left_wheel_joint", "right_wheel_joint", "neck_base_joint",
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "camera_joint", "camera_optical_joint", "rgbd_joint_fixed",
        "rgbd_optical_joint", "lidar_joint",
    }
    assert expected <= joints.keys()
    assert joints["left_wheel_joint"].find("origin").attrib["xyz"] == "0 0.105 0.070"
    assert joints["right_wheel_joint"].find("origin").attrib["xyz"] == "0 -0.105 0.070"
    assert joints["joint6"].find("origin").attrib["xyz"] == "0.068 -0.035 0.036"
    axes = [joints[f"joint{i}"].find("axis").attrib["xyz"] for i in range(1, 7)]
    assert axes == ["0 0 1", "0 1 0", "0 1 0", "0 0 1", "0 1 0", "0 0 1"]
    for index in range(1, 7):
        assert joints[f"joint{index}"].find("origin").attrib["rpy"] == "0 0 0"

    lidar_joint = joints["lidar_joint"]
    assert lidar_joint.attrib["type"] == "fixed"
    assert lidar_joint.find("parent").attrib["link"] == "base_link"
    assert lidar_joint.find("child").attrib["link"] == "lidar_link"
    assert lidar_joint.find("origin").attrib == {
        "xyz": "0.190 0 0.274",
        "rpy": "0 0.898844565 0",
    }


def test_gazebo_lidar_and_point_cloud_bridge_match_sensor_contract():
    root = _expanded_description("gazebo")
    lidar_link = root.find(".//link[@name='lidar_link']")
    assert lidar_link is not None
    lidar_mesh = lidar_link.find("visual/geometry/mesh")
    assert lidar_mesh is not None
    assert lidar_mesh.attrib["filename"].endswith("/meshes/jarvis_lidar.stl")
    assert lidar_mesh.attrib["scale"] == "0.001 0.001 0.001"
    assert (PKG / "meshes/jarvis_lidar.stl").is_file()

    sensor = root.find(".//gazebo[@reference='lidar_link']/sensor")
    assert sensor is not None
    assert sensor.attrib == {"name": "lidar_sensor", "type": "gpu_lidar"}
    assert sensor.findtext("pose") == "0 0 0 0 0 0"
    assert sensor.findtext("topic") == "/lidar"
    assert sensor.findtext("gz_frame_id") == "lidar_link"
    assert float(sensor.findtext("update_rate")) == 15.0
    assert sensor.findtext("always_on") == "true"

    horizontal = sensor.find("lidar/scan/horizontal")
    vertical = sensor.find("lidar/scan/vertical")
    scan_range = sensor.find("lidar/range")
    assert horizontal is not None
    assert vertical is not None
    assert scan_range is not None
    assert [float(horizontal.findtext(tag)) for tag in (
        "samples", "resolution", "min_angle", "max_angle"
    )] == [360.0, 1.0, -3.1415927, 3.1415927]
    assert [float(vertical.findtext(tag)) for tag in (
        "samples", "resolution", "min_angle", "max_angle"
    )] == [96.0, 1.0, 0.0, 1.0471975512]
    assert [float(scan_range.findtext(tag)) for tag in (
        "min", "max", "resolution"
    )] == [0.12, 50.0, 0.01]

    launch_source = (PKG / "launch/jvs_mock_gazebo.launch.py").read_text()
    point_cloud_bridge = (
        "/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked"
    )
    assert point_cloud_bridge in launch_source
    assert "/lidar@sensor_msgs/msg/LaserScan" not in launch_source

    world = ET.parse(PKG / "worlds/empty.world").getroot()
    sensors_system = world.find(
        ".//plugin[@filename='gz-sim-sensors-system']"
    )
    assert sensors_system is not None
    assert sensors_system.findtext("render_engine") == "ogre2"


def test_controller_joint_names_and_dimensions_match_description():
    config = yaml.safe_load((PKG / "config" / "jarvis_controllers.yaml").read_text())
    base = config["base_controller"]["ros__parameters"]
    arm = config["arm_position_controller"]["ros__parameters"]
    assert base["left_wheel_names"] == ["left_wheel_joint"]
    assert base["right_wheel_names"] == ["right_wheel_joint"]
    assert base["wheel_separation"] == 0.21
    assert base["wheel_radius"] == 0.07
    assert base["cmd_vel_timeout"] == 0.5
    assert arm["joints"] == [f"joint{i}" for i in range(1, 7)]
    assert arm["interface_name"] == "position"


def test_joint_space_commands_are_full_and_within_safe_amplitudes():
    motion = _load_motion_module()
    for step in range(201):
        values = motion.sample_joint_positions(step * 0.05)
        assert len(values) == 6
        for value, amplitude in zip(values, motion.SAFE_AMPLITUDES):
            assert math.isfinite(value)
            assert abs(value) <= amplitude + 1e-12


def test_default_relative_task_trajectory_is_10_hz_and_ik_reachable():
    kinematics = _load_kinematics_module()
    trajectory = kinematics.load_relative_trajectory(
        PKG / "trajectories" / "default_task_space_trajectory.json"
    )
    assert trajectory.sample_rate_hz == 10.0
    assert trajectory.duration_s == 6.0
    assert len(trajectory.samples) == 61
    assert all(
        math.isclose(sample.time_from_start, index * 0.1, abs_tol=1e-12)
        for index, sample in enumerate(trajectory.samples)
    )

    initial_q = np.zeros(6)
    initial_position, initial_orientation = kinematics.forward_kinematics(initial_q)
    solutions = []
    seed = initial_q
    for sample in trajectory.samples:
        target = kinematics.relative_pose_target(
            initial_position,
            initial_orientation,
            sample.pose,
            trajectory.frame_id,
        )
        seed = kinematics.solve_ik(*target, seed)
        assert seed is not None
        solutions.append(seed)

    solutions = np.asarray(solutions)
    assert np.all(solutions >= kinematics.JOINT_LOWER)
    assert np.all(solutions <= kinematics.JOINT_UPPER)
    max_speed = np.max(np.abs(np.diff(solutions, axis=0))) * trajectory.sample_rate_hz
    assert max_speed < 1.0

    relative_positions = np.asarray([sample.pose[:3] for sample in trajectory.samples])
    # the default trajectory is a curved camera motion (30 cm reach + 20° of
    # pitch, commit 8bc9498), NOT a straight line — assert the reach/shape
    # contract instead of collinearity
    total_reach = float(np.linalg.norm(relative_positions[-1] - relative_positions[0]))
    assert 0.25 <= total_reach <= 0.35, f"reach {total_reach:.3f} m out of band"
    chord_dev = np.abs(
        relative_positions
        - np.linspace(relative_positions[0], relative_positions[-1], len(relative_positions))
    ).max()
    assert chord_dev > 0.005, "trajectory unexpectedly became a straight line"


def _expanded_description(sim_backend="gazebo"):
    import pytest
    xacro = pytest.importorskip("xacro")
    source = (PKG / "urdf" / "jvs_mock.urdf.xacro").read_text()
    # Resolve the source package without requiring a colcon installation.
    document = xacro.parse(source.replace("$(find jvs_mock)", str(PKG)))
    xacro.process_doc(document, mappings={"sim_backend": sim_backend})
    return ET.fromstring(document.toxml())


def test_simulator_backends_have_separate_launch_contracts():
    launch = PKG / "launch"
    assert (launch / "jvs_mock_gazebo.launch.py").is_file()
    assert (launch / "jvs_mock_isaac.launch.py").is_file()
    assert not (launch / "jvs_mock_sim.launch.py").exists()

    gazebo = _expanded_description("gazebo")
    isaac = _expanded_description("isaac")
    assert gazebo.find("ros2_control") is not None
    assert gazebo.find(".//plugin[@filename='gz_ros2_control-system']") is not None
    assert isaac.find("ros2_control") is None
    assert isaac.find(".//gazebo") is None
    gazebo_caster = gazebo.find(".//joint[@name='front_caster_swivel_joint']/origin")
    isaac_caster = isaac.find(".//joint[@name='front_caster_swivel_joint']/origin")
    assert gazebo_caster is not None
    assert isaac_caster is not None
    assert np.allclose(
        np.fromstring(gazebo_caster.attrib["xyz"], sep=" "),
        (0.180, 0.0, 0.055),
    )
    assert np.allclose(
        np.fromstring(isaac_caster.attrib["xyz"], sep=" "),
        (0.180, 0.0, 0.055),
    )
    for tag in ("link", "joint"):
        assert [ET.tostring(element) for element in gazebo.findall(tag)] == [
            ET.tostring(element) for element in isaac.findall(tag)
        ]


def test_isaac_base_limits_match_gazebo():
    gazebo = yaml.safe_load((PKG / "config/jarvis_controllers.yaml").read_text())
    isaac = yaml.safe_load((PKG / "config/isaac_sim.yaml").read_text())[
        "jvs_isaac_command_adapter"]["ros__parameters"]
    base = gazebo["base_controller"]["ros__parameters"]
    for key in ("wheel_radius", "wheel_separation"):
        assert isaac[key] == base[key]
    assert isaac["publish_rate_hz"] == gazebo["controller_manager"]["ros__parameters"]["update_rate"]
    assert isaac["cmd_vel_timeout_s"] == base["cmd_vel_timeout"]
    for component, axis in (("linear", "linear.x"), ("angular", "angular.z")):
        for limit in ("velocity", "acceleration"):
            assert isaac[f"max_{component}_{limit}"] == base[f"{axis}.max_{limit}"]


def test_rebalanced_masses_have_consistent_inertia():
    import pytest

    root = _expanded_description()
    links = {link.get("name"): link for link in root.findall("link")}
    base = links["base_link"]
    base_mass = 31.772322
    assert float(base.find("inertial/mass").get("value")) == pytest.approx(base_mass)
    mesh = base.find("collision/geometry/mesh")
    assert mesh is not None
    assert mesh.get("filename").endswith("/meshes/base_frustum.stl")
    bottom_radius, top_radius, length = 0.18, 0.13, 0.12
    vertices = np.asarray([
        [float(value) for value in line.split()[1:]]
        for line in (PKG / "meshes/base_frustum.stl").read_text().splitlines()
        if line.strip().startswith("vertex ")
    ])
    assert (vertices[:, 2].min(), vertices[:, 2].max()) == pytest.approx((-0.06, 0.06))
    bottom = vertices[np.isclose(vertices[:, 2], -0.06)]
    top = vertices[np.isclose(vertices[:, 2], 0.06)]
    assert np.hypot(bottom[:, 0], bottom[:, 1]).max() == pytest.approx(bottom_radius)
    assert np.hypot(top[:, 0], top[:, 1]).max() == pytest.approx(top_radius)
    s = bottom_radius**2 + bottom_radius * top_radius + top_radius**2
    q = sum((bottom_radius**(4 - power)) * (top_radius**power) for power in range(5))
    u_bar = (
        bottom_radius**2 + 2 * bottom_radius * top_radius + 3 * top_radius**2
    ) / (4 * s)
    u2 = (
        bottom_radius**2 + 3 * bottom_radius * top_radius + 6 * top_radius**2
    ) / (10 * s)
    inertia_origin_z = float(base.find("inertial/origin").get("xyz").split()[2])
    assert inertia_origin_z == pytest.approx(0.13 - length / 2 + length * u_bar)
    base_inertia = base.find("inertial/inertia")
    assert float(base_inertia.get("ixx")) == pytest.approx(
        base_mass * (3 * q / (20 * s) + length**2 * (u2 - u_bar**2))
    )
    assert float(base_inertia.get("iyy")) == pytest.approx(
        float(base_inertia.get("ixx"))
    )
    assert float(base_inertia.get("izz")) == pytest.approx(
        base_mass * 3 * q / (10 * s)
    )

    expected_masses = {"neck_base_link": 0.18}
    expected_masses.update(zip(
        (f"link{index}" for index in range(1, 7)),
        (0.168, 0.21, 0.192, 0.144, 0.30, 0.108),
    ))
    for name, mass in expected_masses.items():
        link = links[name]
        assert float(link.find("inertial/mass").get("value")) == pytest.approx(mass)
        cylinder = link.find("collision/geometry/cylinder")
        radius = float(cylinder.get("radius"))
        length = float(cylinder.get("length"))
        inertia = link.find("inertial/inertia")
        assert float(inertia.get("ixx")) == pytest.approx(mass * (3 * radius**2 + length**2) / 12)
        assert float(inertia.get("iyy")) == pytest.approx(float(inertia.get("ixx")))
        assert float(inertia.get("izz")) == pytest.approx(mass * radius**2 / 2)


def test_isaac_acceleration_limits_start_reverse_and_stop():
    import pytest

    bridge = _load_isaac_bridge_module()
    limit = bridge.acceleration_limited_twist
    assert limit((1.0, 3.0), (0.0, 0.0), 0.01, 1.0, 7.0) == pytest.approx((0.01, 0.07))
    assert limit((-1.0, -3.0), (0.5, 1.0), 0.1, 1.0, 7.0) == pytest.approx((0.4, 0.3))
    assert limit((0.0, 0.0), (0.05, -0.1), 0.1, 1.0, 7.0) == (0.0, 0.0)
    assert limit((1.0, 3.0), (0.0, 0.0), 0.0, 1.0, 7.0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        limit((0.0, 0.0), (0.0, 0.0), 0.01, -1.0, 7.0)


def test_isaac_command_contract_validation():
    import pytest

    bridge = _load_isaac_bridge_module()
    values = [0.1 * index for index in range(6)]
    assert bridge.arm_positions(values) == values
    with pytest.raises(ValueError, match="expected 6"):
        bridge.arm_positions(values[:-1])
    with pytest.raises(ValueError, match="non-finite"):
        bridge.arm_positions([0.0] * 5 + [float("nan")])
    assert bridge.limited_twist(2.0, -4.0, 1.0, 3.0) == (1.0, -3.0)
    assert bridge.differential_wheel_velocities(
        0.7, 0.0, 0.07, 0.21
    ) == pytest.approx([10.0, 10.0])
    assert bridge.differential_wheel_velocities(
        0.0, 1.0, 0.07, 0.14
    ) == pytest.approx([-1.0, 1.0])


def test_passive_casters_match_target_footprint_and_touch_ground():
    import pytest
    import shutil
    import subprocess
    import tempfile

    root = _expanded_description()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    expected = {
        "front_caster": (0.180, 0.0),
        "rear_left_caster": (-0.131, 0.089),
        "rear_right_caster": (-0.131, -0.089),
    }
    contacts = []
    for name, xy in expected.items():
        swivel = joints[f"{name}_swivel_joint"]
        wheel_joint = joints[f"{name}_wheel_joint"]
        swivel_link = links[f"{name}_swivel_link"]
        wheel = links[f"{name}_wheel_link"]
        assert swivel.get("type") == "continuous"
        assert wheel_joint.get("type") == "continuous"
        assert swivel.find("axis").get("xyz") == "0 0 1"
        assert wheel_joint.find("axis").get("xyz") == "0 1 0"
        swivel_xyz = np.fromstring(swivel.find("origin").get("xyz"), sep=" ")
        wheel_xyz = np.fromstring(wheel_joint.find("origin").get("xyz"), sep=" ")
        center = swivel_xyz + wheel_xyz
        radius = float(wheel.find("collision/geometry/cylinder").get("radius"))
        assert tuple(center[:2]) == pytest.approx(xy)
        assert math.isclose(center[2] - radius, 0.0, abs_tol=1e-9)
        post = swivel_link.find("collision")
        post_origin_z = float(post.find("origin").get("xyz").split()[2])
        post_length = float(post.find("geometry/cylinder").get("length"))
        post_radius = float(post.find("geometry/cylinder").get("radius"))
        post_top = swivel_xyz[2] + post_origin_z + post_length / 2
        assert post_top > 0.07
        # At the post top, the tapered body and post overlap horizontally.
        body_radius = 0.18 + (0.13 - 0.18) * ((post_top - 0.07) / 0.12)
        assert np.hypot(*xy) < body_radius + post_radius
        contacts.append(center[:2])
    assert len(contacts) == 3
    if shutil.which("ign"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "robot.urdf"
            path.write_text(ET.tostring(root, encoding="unicode"))
            result = subprocess.run(["ign", "sdf", "-p", str(path)],
                                    capture_output=True, text=True, check=True)
        sdf = ET.fromstring(result.stdout)
        for name in expected:
            link = sdf.find(f".//link[@name='{name}_wheel_link']")
            assert link is not None
            collision = link.find("collision")
            assert collision.find("geometry/cylinder") is not None
            assert float(collision.find("surface/friction/ode/mu").text) == 0.02
            assert float(collision.find("surface/friction/ode/mu2").text) == 0.02


def test_sampled_arm_poses_have_static_support_margin():
    """Mass-weighted COM must stay inside the ground-contact polygon.

    Includes every joint-limit corner plus deterministic interior samples;
    this is a static, level-floor check, not a dynamic tipping guarantee.
    """
    import itertools

    root = _expanded_description()
    kinematics = _load_kinematics_module()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    polygon = np.array([
        [-.131, -.089], [0, -.105], [.180, 0], [0, .105], [-.131, .089],
    ])
    corners = np.array(list(itertools.product(*zip(
        kinematics.JOINT_LOWER, kinematics.JOINT_UPPER))))
    samples = np.random.default_rng(42).uniform(
        kinematics.JOINT_LOWER, kinematics.JOINT_UPPER, (256, 6))
    for q in np.vstack((corners, samples, np.zeros((1, 6)))):
        transforms = {"base_link": np.eye(4)}
        pending = list(joints)
        while pending:
            for joint in pending[:]:
                parent = joint.find("parent").get("link")
                if parent not in transforms:
                    continue
                origin = joint.find("origin")
                local = np.eye(4)
                local[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
                local[:3, :3] = kinematics.rpy_matrix(
                    *np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
                if joint.get("name") in kinematics.JOINT_NAMES:
                    i = kinematics.JOINT_NAMES.index(joint.get("name"))
                    local[:3, :3] = local[:3, :3] @ kinematics._rotation(kinematics.JOINT_AXES[i], q[i])
                transforms[joint.find("child").get("link")] = transforms[parent] @ local
                pending.remove(joint)
        weighted = np.zeros(3)
        total_mass = 0.0
        for name, link in links.items():
            inertial = link.find("inertial")
            if inertial is None:
                continue
            mass = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            center = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz"), sep=" ")
            weighted += mass * (transforms[name] @ np.r_[center, 1])[:3]
            total_mass += mass
        center = weighted[:2] / total_mass
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            relative = center - start
            margin = (edge[0] * relative[1] - edge[1] * relative[0]) / np.linalg.norm(edge)
            assert margin > 0.025, (q, center, margin)
