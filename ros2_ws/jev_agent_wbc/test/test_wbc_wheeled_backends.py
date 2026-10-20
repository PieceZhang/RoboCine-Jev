"""Wheeled OCS2 model contracts; optional genuine native solver subprocesses."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ["Robot_Husky_UR5", "Robot_Ridgeback_UR5"]


@pytest.fixture(params=ROBOTS)
def adapter(request):
    path = ROOT / request.param / "ocs2/adapter.py"
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WheeledBackendAdapter()


def test_authored_camera_closure(adapter):
    actual = adapter.measured_camera_pose(
        adapter.model["T_world_base_at_authored_pose"], adapter.q_rest
    )
    np.testing.assert_allclose(actual, adapter.model["T_world_camera_at_authored_pose"], atol=2e-7)


def test_usd_and_sensor_model_are_current(adapter):
    for path_key, hash_key in [("model", "model_sha256"), ("sensor_config", "sensor_config_sha256")]:
        source = adapter.folder.parent / adapter.model[path_key]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == adapter.model[hash_key]


def test_fk_derivative_matches_authored_joint_screws(adapter):
    """An independent spatial-screw check on actual optical camera motion."""
    nominal = adapter.camera_in_base(adapter.q_rest)
    for index, joint in enumerate(adapter.model["joints"]):
        plus = adapter.q_rest.copy()
        minus = plus.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        a, b = adapter.camera_in_base(plus), adapter.camera_in_base(minus)
        omega, velocity = np.array(joint["screw_omega"]), np.array(joint["screw_v"])
        np.testing.assert_allclose(
            (a[:3, 3] - b[:3, 3]) / 2e-6,
            np.cross(omega, nominal[:3, 3]) + velocity, atol=2e-7,
        )
        np.testing.assert_allclose(
            Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec() / 2e-6,
            omega, atol=2e-7,
        )


def test_full_measured_base_orientation_and_height_are_retained(adapter):
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [.12, -.08, .71]).as_matrix()
    base[:3, 3] = [.4, -.7, .23]
    q = np.array(adapter.spec["initial_arm"])
    expected = base @ adapter.camera_in_base(q)
    np.testing.assert_allclose(adapter.measured_camera_pose(base, q), expected, atol=1e-12)
    joints = dict(zip(adapter.joint_names, q))
    state = adapter.state_from_measured(base, joints)
    assert state[2] == pytest.approx(.71)
    assert np.linalg.norm(expected - adapter.camera_pose(state)) > .01


@pytest.mark.parametrize("frame", ["world", "map"])
def test_reference_conversion_keeps_optical_waypoint(adapter, frame):
    pose, world_map = np.eye(4), np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [.2, -.4, .9]).as_matrix()
    pose[:3, 3] = [1., 2., .8]
    world_map[:3, :3] = Rotation.from_euler("z", .63).as_matrix()
    world_map[:3, 3] = [-2., .1, .5]
    original = pose.copy()
    tool = adapter.tool_reference(pose, frame, world_map)
    expected = pose if frame == "world" else world_map @ pose
    np.testing.assert_allclose(tool @ adapter.T_ee_camera, expected, atol=1e-12)
    np.testing.assert_array_equal(pose, original)


def test_unresolved_reference_frame_rejected(adapter):
    with pytest.raises(ValueError):
        adapter.tool_reference(np.eye(4), "map")
    with pytest.raises(ValueError):
        adapter.tool_reference(np.eye(4), "base")


@pytest.mark.parametrize("yaw", [-2.3, 0., .7, 1.5707963267948966])
def test_actual_base_model_and_actuator_frames(adapter, yaw):
    state = np.zeros(9)
    state[2] = yaw
    control = np.zeros(adapter.spec["input_dim"])
    control[0] = .17
    control[1] = -.08
    if adapter.omni:
        control[2] = .12
    derivative = adapter.flow_map(state, control)
    body_lateral = -np.sin(yaw) * derivative[0] + np.cos(yaw) * derivative[1]
    command = adapter.actuators(state, control)
    if adapter.omni:
        assert body_lateral == pytest.approx(-.08)
        np.testing.assert_allclose(command["velocities"][:3], derivative[:3])
        assert command["wheel_ground_dynamics_verified"] is False
    else:
        assert body_lateral == pytest.approx(0., abs=1e-15)
        profile = json.loads((adapter.folder.parent / "robot.json").read_text())
        left, right = command["wheel_velocities"][0], command["wheel_velocities"][-1]
        assert (right + left) * profile["wheel_radius_m"] / 2 == pytest.approx(control[0])
        assert (right - left) * profile["wheel_radius_m"] / profile["wheel_separation_m"] == pytest.approx(control[1])
        assert command["skid_slip_compensated"] is False


def test_nonfinite_and_wrong_controls_rejected(adapter):
    with pytest.raises(ValueError):
        adapter.flow_map(np.zeros(9), np.zeros(10))
    with pytest.raises(ValueError):
        adapter.flow_map(np.zeros(9), np.full(adapter.spec["input_dim"], np.nan))
    with pytest.raises(ValueError):
        adapter.camera_pose(np.full(9, np.inf))


def test_unsupported_physical_and_geometry_requests_fail_closed(adapter):
    for requests in [(), [{"minimum_surface_clearance_m": .2}]]:
        response = adapter.negotiate(requests)
        assert response["supported"] is False
        assert response["feasible"] is None
        assert response["reason"]


@pytest.mark.parametrize("times", [[0., 0.], [1., 2.], [0., float("nan")], [0., 11.]])
def test_invalid_timed_requests_rejected(adapter, times):
    joints = dict(zip(adapter.joint_names, adapter.spec["initial_arm"]))
    with pytest.raises(ValueError):
        adapter.native_request(np.eye(4), joints, times, [np.eye(4), np.eye(4)])


def test_native_request_keeps_all_fixed_camera_samples(adapter):
    joints = dict(zip(adapter.joint_names, adapter.spec["initial_arm"]))
    initial = adapter.measured_camera_pose(np.eye(4), adapter.spec["initial_arm"])
    final = initial.copy()
    final[:3, 3] += [.03, .04, .02]
    result = adapter.native_request(np.eye(4), joints, [0., .6], [initial, final])
    np.testing.assert_allclose(result["camera_poses"], [initial, final], atol=1e-12)
    assert result["timestamps_s"] == [0., .6]
    assert result["physical_commands_issued"] is False


def test_genuine_native_mpc_request_and_policy(adapter, tmp_path):
    name = "JEV_OCS2_BIN_RIDGEBACK" if adapter.omni else "JEV_OCS2_BIN_HUSKY"
    binary = os.environ.get(name)
    if not binary:
        pytest.skip(f"set {name} to the genuine compiled camera_mpc_probe")
    q = adapter.spec["initial_arm"]
    measured_base = np.eye(4)
    measured_base[2, 3] = adapter.nominal_height
    start = adapter.measured_camera_pose(measured_base, q)
    goal = start.copy()
    goal[:3, 3] += [.06, .08, .04]
    goal[:3, :3] = Rotation.from_rotvec(.06 * np.array([.4, .5, -.6]) / np.linalg.norm([.4, .5, -.6])).as_matrix() @ start[:3, :3]
    request = adapter.native_request(measured_base, dict(zip(adapter.joint_names, q)), [0., 1.], [start, goal])
    request_path, policy_path = tmp_path / "request.json", tmp_path / "policy.json"
    request_path.write_text(json.dumps(request))
    process = subprocess.run([binary, str(adapter.folder / "backend.json"), str(adapter.folder / "model.json"), str(request_path), str(policy_path)], text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    assert result["mpc_executed"] is True
    assert result["physical_execution_verified"] is False
    assert result["input_dim"] == adapter.spec["input_dim"]
    assert result["maximum_speed_ratio"] <= 1
    policy = json.loads(policy_path.read_text())
    assert policy["physical_commands_issued"] is False
    assert len(policy["states"]) == len(policy["controls"]) == len(policy["timestamps_s"])
    assert len(policy["states"]) > 2
    np.testing.assert_allclose(policy["states"][0], request["initial_state"], atol=1e-12)
    np.testing.assert_allclose(adapter.camera_pose(policy["states"][-1]), goal, atol=.003)


def test_native_unreachable_fixed_reference_is_rejected(adapter, tmp_path):
    """The solver must not certify an unreachable path by rewriting its target."""
    name = "JEV_OCS2_BIN_RIDGEBACK" if adapter.omni else "JEV_OCS2_BIN_HUSKY"
    binary = os.environ.get(name)
    if not binary:
        pytest.skip(f"set {name} to the genuine compiled camera_mpc_probe")
    q = adapter.spec["initial_arm"]
    base = np.eye(4)
    base[2, 3] = adapter.nominal_height
    start = adapter.measured_camera_pose(base, q)
    goal = start.copy()
    goal[:3, 3] += [8., 0., 0.]
    request = adapter.native_request(base, dict(zip(adapter.joint_names, q)), [0., 1.], [start, goal])
    request_path, policy_path = tmp_path / "unreachable.json", tmp_path / "policy.json"
    request_path.write_text(json.dumps(request))
    process = subprocess.run([binary, str(adapter.folder / "backend.json"), str(adapter.folder / "model.json"), str(request_path), str(policy_path)], text=True, capture_output=True, timeout=30)
    assert process.returncode != 0
    assert not policy_path.exists()
    if process.stdout:
        assert json.loads(process.stdout)["passed"] is False


def safety_module():
    path=ROOT/'scripts/ocs2_wheeled_safety.py'
    spec=importlib.util.spec_from_file_location('wheeled_safety_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('omni',[False,True])
def test_hard_command_acceleration_and_original_joint_braking(omni):
    safety=safety_module();guard=safety.HardCommandGuard(omni,np.full(6,-2.),np.full(6,2.))
    for _ in range(40):
        old=guard.previous.copy();u=guard.command(np.full(guard.nu,10.),np.zeros(6))
        assert np.all(np.abs(u)<=guard.speed+1e-12)
        assert np.all(np.abs(u-old)<=guard.acceleration*.01+1e-12)
    for _ in range(100):u=guard.command(np.zeros(guard.nu),np.zeros(6),brake=True)
    np.testing.assert_allclose(u,0.,atol=1e-12)
    guard.previous[-1]=.6
    with pytest.raises(RuntimeError,match='braking envelope'):
        guard.command(np.ones(guard.nu),np.r_[np.zeros(5),1.999])


def test_native_policy_absolute_time_and_strict_lease():
    safety=safety_module();states=np.zeros((2,17));states[1,9:]=.1
    policy=dict(ok=True,state_stamp_s=10.,times_s=[10.,11.],states=states.tolist())
    np.testing.assert_allclose(safety.policy_velocity(policy,10.1,8),.01,atol=1e-12)
    for t in [9.999,10.201,12.]:
        with pytest.raises(ValueError,match='expired|future'):
            safety.policy_velocity(policy,t,8)


def test_convex_narrow_phase_rejects_bbox_false_overlap_and_real_contact():
    safety=safety_module()
    a=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    # The hull AABBs overlap; the convex tetrahedra are still strictly separate.
    b=1.1-a
    lower=safety.convex_separation(a,b)
    assert .7<lower<.8
    assert safety.convex_separation(a,a+.05)==pytest.approx(0.,abs=1e-8)


def test_collision_fixed_intermediate_without_geometry_is_collapsed():
    safety=safety_module()
    shapes=[dict(name=n,body=n,center_body=[0.,0.,0.],half_extents_body=[1.,1.,1.]) for n in ['base','fixed_tip','arm']]
    guard=safety.CollisionGuard(shapes,[('base','no_collision_middle',True),('no_collision_middle','fixed_tip',True),('base','arm',False)])
    assert not guard.pairs


def test_actual_collision_hull_native_abi_preserves_world_se3(adapter):
    """Independent rigid-frame invariance check on the actual robot hulls."""
    safety=safety_module();path=adapter.folder/'evidence/live_model_static.json'
    library=adapter.folder/'.cache/backend/libmeasured_collision_guard.so'
    if not path.is_file() or not library.is_file():
        pytest.skip('build native measured_collision_guard and inspect actual USD geometry')
    from scipy.linalg import expm
    model=json.loads(path.read_text());prefix=[np.eye(4)]
    for q,j in zip(adapter.spec['initial_arm'],model['joints']):
        w=np.array(j['screw_omega']);hat=np.zeros((4,4));hat[:3,:3]=[[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]];hat[:3,3]=j['screw_v']
        prefix.append(prefix[-1]@expm(hat*(q-j['q_at_authored_pose'])))
    bodies={b['name']:prefix[b['prefix']]@np.array(b['T_base_body']) for b in model['body_kinematics']}
    guard=safety.NativeCollisionGuard(library,path)
    try:
        nominal=guard.check(bodies)[0]
        assert nominal>.005
        transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[.6,-.4,1.2]).as_matrix();transform[:3,3]=[2.,-.7,.8]
        moved=guard.check({name:transform@pose for name,pose in bodies.items()})[0]
        assert moved==pytest.approx(nominal,abs=2e-8)
        with pytest.raises((KeyError,ValueError)):
            guard.check({})
    finally:guard.close()
