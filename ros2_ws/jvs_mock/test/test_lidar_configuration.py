import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


PACKAGE = Path(__file__).resolve().parents[1]
PROFILE_PATH = PACKAGE / "config" / "JVS_Nearfield.json"
ROBOT_XACRO = PACKAGE / "urdf" / "jvs_mock.urdf.xacro"
GAZEBO_XACRO = PACKAGE / "urdf" / "gazebo.xacro"


def test_simview_cli_parses_before_isaac_starts():
    result = subprocess.run(
        [sys.executable, str(PACKAGE / "scripts" / "jvs_simview_isaac.py"), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--lidar-config" in result.stdout


def _profile():
    return json.loads(PROFILE_PATH.read_text())["profile"]


def test_nearfield_profile_has_full_rotary_coverage():
    profile = _profile()
    state = profile["emitterStates"][0]
    count = profile["numberOfEmitters"]

    assert profile["scanType"] == "rotary"
    assert profile["nearRangeM"] <= 0.10
    assert profile["farRangeM"] >= 3.0
    assert profile["scanRateBaseHz"] == 10.0
    assert count >= 64
    assert len(state["azimuthDeg"]) == count
    assert len(state["elevationDeg"]) == count
    assert len(state["fireTimeNs"]) == count
    assert min(state["elevationDeg"]) <= -45.0
    assert max(state["elevationDeg"]) >= 45.0


def test_robot_preserves_rear_and_restores_original_front_lidar():
    robot = ET.parse(ROBOT_XACRO).getroot()
    links = robot.findall("./link[@name='lidar_link']")
    joints = robot.findall("./joint[@name='lidar_joint']")

    assert len(links) == 1
    assert len(joints) == 1
    origin = joints[0].find("origin")
    xyz = tuple(float(value) for value in origin.attrib["xyz"].split())
    rpy = tuple(float(value) for value in origin.attrib["rpy"].split())

    assert xyz == (-0.085, 0.0, 0.235)
    assert rpy == (0.0, 0.0, 0.0)
    assert abs(xyz[0]) <= 0.13
    assert xyz[2] > 0.19
    front = robot.find("./joint[@name='front_lidar_joint']/origin")
    assert tuple(map(float, front.attrib['xyz'].split())) == (.190, 0., .274)
    assert tuple(map(float, front.attrib['rpy'].split())) == (0., .898844565, 0.)
    assert len(robot.findall("./link[@name='front_lidar_link']")) == 1
    assert len([link for link in robot.findall('link') if 'lidar' in link.attrib['name']]) == 2


def test_gazebo_matches_packaged_nearfield_profile():
    gazebo = ET.parse(GAZEBO_XACRO).getroot()
    sensors = gazebo.findall(".//sensor[@type='gpu_lidar']")
    assert len(sensors) == 2
    assert sensors[1].findtext('gz_frame_id') == 'front_lidar_link'
    assert sensors[1].findtext('topic') == '/lidar/front'
    assert sensors[1].findtext('lidar/scan/vertical/samples') == '96'

    vertical = sensors[0].find("./lidar/scan/vertical")
    range_config = sensors[0].find("./lidar/range")
    profile = _profile()
    elevations = profile["emitterStates"][0]["elevationDeg"]

    assert int(vertical.findtext("samples")) == profile["numberOfEmitters"]
    assert math.isclose(
        float(vertical.findtext("min_angle")), math.radians(min(elevations)), abs_tol=1e-9
    )
    assert math.isclose(
        float(vertical.findtext("max_angle")), math.radians(max(elevations)), abs_tol=1e-9
    )
    assert float(range_config.findtext("min")) <= profile["nearRangeM"] + 0.02
