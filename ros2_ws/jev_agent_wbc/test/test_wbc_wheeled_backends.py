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


def current_camera_collision_fixture(adapter, tmp_path):
    """Reuse fixed physical hulls with the current calibrated optical mount.

    Historical evidence remains immutable. Mixing today's camera target with
    the old optical frame would instead test an unintended 180-degree move.
    """
    source=adapter.folder/'evidence/live_model_static.json'
    if not source.is_file():
        pytest.skip('export the actual native collision geometry first')
    document=json.loads(source.read_text())
    for key in ('joints','T_base_ee_at_authored_pose'):
        assert document[key]==adapter.model[key], 'physical kinematics require a new collision export'
    for key in ('T_ee_camera_optical','T_world_camera_at_authored_pose','sensor_config_sha256'):
        document[key]=adapter.model[key]
    destination=tmp_path/'current_camera_collision_model.json'
    destination.write_text(json.dumps(document))
    return destination


def native_worker(folder):
    return folder/'.cache'/os.environ.get('JEV_OCS2_WORKER_CACHE','backend')/'camera_mpc_worker'


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
        old=guard.previous.copy();u=guard.command(np.full(guard.nu,10.),np.zeros(6),np.zeros(6))
        assert np.all(np.abs(u)<=guard.speed+1e-12)
        assert np.all(np.abs(u-old)<=guard.acceleration*.01+1e-12)
    for _ in range(100):u=guard.command(np.zeros(guard.nu),np.zeros(6),np.zeros(6),brake=True)
    np.testing.assert_allclose(u,0.,atol=1e-12)
    guard.previous[-1]=.6
    with pytest.raises(RuntimeError,match='braking envelope'):
        guard.command(np.ones(guard.nu),np.r_[np.zeros(5),1.999],np.zeros(6))


@pytest.mark.parametrize('omni',[False,True])
def test_measured_velocity_and_braking_audit_do_not_inherit_command_state(omni):
    safety=safety_module();guard=safety.HardCommandGuard(omni,np.full(6,-2.),np.full(6,2.))
    # Command is zero, but the real drive still moves fast near its upper limit.
    u=guard.command(np.ones(guard.nu),np.r_[np.zeros(5),1.95],np.r_[np.zeros(5),.6])
    np.testing.assert_array_equal(u,np.zeros(guard.nu))
    assert not guard.last_audit['actual_braking_ok']
    assert guard.last_audit['minimum_actual_braking_margin_rad']<0
    guard.command(np.ones(guard.nu),np.zeros(6),np.full(6,.7))
    assert not guard.last_audit['actual_velocity_ok']
    assert guard.last_audit['maximum_actual_velocity_ratio']>1
    guard.command(np.ones(guard.nu),np.r_[np.zeros(5),2.001],np.zeros(6))
    assert not guard.last_audit['configuration_ok']
    # The emergency tolerance never turns the original-bound violation into pass.
    with pytest.raises(RuntimeError,match='original bound'):
        guard.command(np.zeros(guard.nu),np.full(6,2.016),np.zeros(6))


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


def test_persistent_native_slq_recovers_then_preserves_absolute_time_and_hard_caps(adapter,tmp_path):
    binary=native_worker(adapter.folder)
    model=current_camera_collision_fixture(adapter,tmp_path)
    if not binary.is_file() or not model.is_file():
        pytest.skip('build the genuine native SLQ worker and actual collision model')
    nu=adapter.spec['input_dim'];q=np.array(adapter.spec['initial_arm']);state=np.r_[np.zeros(3),q,np.zeros(nu)]
    base=np.eye(4);base[2,3]=adapter.nominal_height
    camera=adapter.measured_camera_pose(base,q);goal=camera.copy();goal[:3,3]+=[.04,0,.02]
    goal[:3,:3]=Rotation.from_rotvec([0,.03,0]).as_matrix()@camera[:3,:3]
    request=dict(request_id=1,state_stamp_s=10.,horizon_s=1.,state=state.tolist(),
                 T_world_base=base.tolist(),frame='world_optical',absolute_times_s=[10.,11.],
                 camera_poses=[camera.tolist(),goal.tolist()])
    process=subprocess.Popen([str(binary),str(adapter.folder/'backend.json'),str(model)],
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def solve(row):
        process.stdin.write(json.dumps(row)+'\n');process.stdin.flush()
        return json.loads(process.stdout.readline())
    try:
        # A missing state interpretation is rejected; the same process survives.
        assert solve(request)['ok'] is False
        request['rate_state_semantics']='last_applied_velocity_command'
        first=solve(request)
        assert first['ok'] is True,first
        assert first['framework']=='OCS2_SLQ_input_rate'
        assert first['rate_state_semantics']==request['rate_state_semantics']
        assert first['maximum_velocity_ratio']<=1.00001
        assert first['maximum_acceleration_ratio']<=1.00001
        assert first['minimum_self_clearance_lower_bound_m']>=.005
        assert first['minimum_joint_margin_rad']>=0
        request.update(request_id=2,state_stamp_s=110.,absolute_times_s=[110.,111.])
        shifted=solve(request)
        assert shifted['ok'] is True,shifted
        assert shifted['times_s'][0]==pytest.approx(110.)
        assert shifted['times_s'][-1]==pytest.approx(111.)
        grid=np.linspace(10.,11.,101)
        # Adaptive ODE knots can differ after an absolute-clock shift. Compare
        # the same physical instants rather than requiring identical grids.
        a=np.array(first['states']);b=np.array(shifted['states'])
        for k in range(a.shape[1]):
            np.testing.assert_allclose(np.interp(grid,first['times_s'],a[:,k]),
                np.interp(grid+100.,shifted['times_s'],b[:,k]),atol=1e-3)
        assert process.poll() is None
    finally:
        process.terminate();process.communicate(timeout=3.)


def test_skid_wheel_servo_compensates_measured_slip_with_original_wheel_caps():
    safety=safety_module();servo=safety.SkidSteerVelocityServo(.1651,.5708)
    for i in range(300):
        old=servo.previous.copy();wheel=servo.command([.1,.2],.02)
        assert np.max(np.abs(wheel))<=8.
        assert np.max(np.abs(wheel-old))<=servo.wheel_slew*.01+1e-12
        assert abs(servo.integral)<=.65
    assert servo.last_audit['virtual_actuator_yaw_rate']>.2
    for _ in range(600):
        old=servo.previous.copy();wheel=servo.command([-.22,-.35],.4)
        assert np.max(np.abs(wheel))<=8.
        assert np.max(np.abs(wheel-old))<=servo.wheel_slew*.01+1e-12
        assert abs(servo.last_audit['virtual_actuator_yaw_rate'])<=4.
        assert abs(servo.integral)<=.65
    assert wheel[0]>wheel[1]
    for _ in range(300):wheel=servo.command([0.,0.],.01,brake=True)
    np.testing.assert_allclose(wheel,0.,atol=1e-12)
    assert servo.integral==0.


def test_reference_window_retains_all_original_interpolation_segments():
    safety=safety_module();times=np.arange(10.,31.0001,.05)
    poses=[dict(original_index=i) for i in range(len(times))]
    for stamp in [10.,10.017,18.125,29.98]:
        horizon=min(1.,31.-stamp)
        subset,selected,(first,last)=safety.reference_window(times,poses,stamp,horizon)
        assert selected==poses[first:last]
        np.testing.assert_array_equal(subset,times[first:last])
        assert subset[0]<=stamp and subset[-1]>=stamp+horizon-1e-9
        assert len(subset)<=23
        # Each point in the interval has precisely the same bracketing knots.
        for t in np.linspace(stamp,stamp+horizon,101):
            full=max(0,min(len(times)-2,int(np.searchsorted(times,t,side='right'))-1))
            local=max(0,min(len(subset)-2,int(np.searchsorted(subset,t,side='right'))-1))
            assert full==first+local


def test_camera_wheel_mapping_is_identical_to_frozen_passing_camera03():
    old_path=ROOT/'Robot_Husky_UR5/ocs2/evidence/camera03_safety_source.py'
    spec=importlib.util.spec_from_file_location('passing_camera03_safety',old_path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    previous=old.SkidSteerVelocityServo(.1651,.5708)
    current=safety_module().SkidSteerVelocityServo(.1651,.5708)
    for i in range(1000):
        t=i*.01;u=[.16*np.sin(.3*t),.3*np.sin(.6*t)];measured=.2*np.sin(.6*t-.1);brake=i%200<20
        np.testing.assert_array_equal(current.command(u,measured,brake=brake),previous.command(u,measured,brake=brake))
        assert current.integral==previous.integral


def test_manual_turn_prebraking_hysteresis_reverse_and_mode_changes_preserve_slew():
    servo=safety_module().SkidSteerVelocityServo(.1651,.5708)
    def tick(velocity,actual,turn=True,brake=False):
        before=servo.previous.copy();wheel=servo.command(velocity,actual,turn_in_place=turn,brake=brake)
        assert np.max(np.abs(wheel-before))<=servo.wheel_slew*.01+1e-12
        assert np.max(np.abs(wheel))<=8.
        return wheel
    for _ in range(100):wheel=tick([0,.35],0.)
    assert np.max(np.abs(wheel))<=1.+1e-12
    tick([0,.35],.23);assert servo.last_audit['turn_braking']
    for _ in range(100):wheel=tick([0,.35],.15)
    np.testing.assert_allclose(wheel,0.,atol=1e-12)

    assert servo.integral==0. and servo.last_audit['turn_braking']
    tick([0,.35],.11);assert not servo.last_audit['turn_braking']
    tick([0,-.35],-.25);assert servo.last_audit['turn_braking']
    for _ in range(100):tick([0,-.35],-.11)
    # A transition from unrestricted camera mapping must decelerate through the
    # existing slew bound, rather than jumping instantly to the turn target cap.
    for _ in range(300):wheel=tick([.2,.35],0.,turn=False)
    assert np.max(np.abs(wheel))>1.
    for _ in range(300):wheel=tick([0,-.35],0.,turn=True)
    assert np.max(np.abs(wheel))<=1.+1e-12
    for _ in range(300):wheel=tick([0,0],0.,turn=False,brake=True)
    np.testing.assert_allclose(wheel,0.,atol=1e-12)

def test_terminal_reference_retains_original_knots_and_only_extends_same_endpoint():
    safety=safety_module();times=np.arange(10.,31.0001,.05);poses=[]
    for i in range(len(times)):
        T=np.eye(4);T[0,3]=i*.001;poses.append(T.tolist())
    before=json.dumps(poses)
    for stamp in [10.,29.98,30.75,31.,32.75,35.99]:
        ts,selected,(first,last),hold=safety.terminal_reference_window(times,poses,stamp,1.)
        assert ts[0]<=stamp and ts[-1]>=stamp+1.-1e-9
        np.testing.assert_array_equal(ts[:last-first],times[first:last])
        assert selected[:last-first]==poses[first:last]
        for t,p in zip(ts,selected):
            if t>times[-1]+1e-9:assert p==poses[-1]
        assert hold['mode']=='constant_original_endpoint_after_end'
        assert hold['original_end_s']==times[-1]
        assert hold['T_world_original_endpoint']==poses[-1]
    assert json.dumps(poses)==before


def test_rms_pass_cannot_hide_original_task_final_tolerance_failure():
    safety=safety_module()
    report=safety.camera_task_metrics([.01,.035],[.01,.02],.0302413629,.025336,.03,.08)
    assert report['position_rmse_m']<.03
    assert report['position_exceeding_samples']==1
    assert report['position_max_m']==.035
    assert report['final_task_tolerance_pass'] is False
    assert safety.camera_task_metrics([.001],[.001],.006,.002,.005,.01)['final_task_tolerance_pass'] is False
    assert safety.camera_task_metrics([.001],[.001],.004,.002,.005,.01)['final_task_tolerance_pass'] is True
    with pytest.raises(ValueError):safety.camera_task_metrics([.001],[.001],float('nan'),.002,.005,.01)


def test_husky_native_terminal_hold_stops_chassis_and_rejects_changed_endpoint(tmp_path):
    folder=ROOT/'Robot_Husky_UR5/ocs2'
    sp=importlib.util.spec_from_file_location('terminal_husky_adapter',folder/'adapter.py')
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);adapter=module.WheeledBackendAdapter()
    binary=native_worker(folder);model=current_camera_collision_fixture(adapter,tmp_path)
    if not binary.is_file() or not model.is_file():pytest.skip('build genuine native terminal-hold SLQ')
    q=np.array(adapter.spec['initial_arm']);state=np.r_[np.zeros(3),q,np.zeros(8)]
    base=np.eye(4);base[2,3]=adapter.nominal_height
    camera=adapter.measured_camera_pose(base,q);goal=camera.copy();goal[:3,3]+=[.015,.005,0.]
    request=dict(request_id=1,state_stamp_s=10.2,horizon_s=1.,state=state.tolist(),
                 rate_state_semantics='last_applied_velocity_command',T_world_base=base.tolist(),
                 frame='world_optical',absolute_times_s=[10.1,11.2],camera_poses=[goal.tolist(),goal.tolist()],
                 terminal_hold=dict(mode='constant_original_endpoint_after_end',original_end_s=10.1,T_world_original_endpoint=goal.tolist(),chassis_brake_applied=True))
    process=subprocess.Popen([str(binary),str(folder/'backend.json'),str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def solve(row):
        process.stdin.write(json.dumps(row)+'\n');process.stdin.flush();return json.loads(process.stdout.readline())
    try:
        corrupt=json.loads(json.dumps(request));corrupt['camera_poses'][-1][0][3]+=.02
        assert solve(corrupt)['ok'] is False
        result=solve(request);assert result['ok'] is True,result
        states=np.array(result['states'])
        assert result['chassis_brake_mode']=='fixed_measured_base'
        assert np.max(np.abs(states[:,9:11]))<1e-10
        np.testing.assert_allclose(states[:,:3],np.broadcast_to(states[0,:3],states[:,:3].shape),atol=1e-10)
        final=adapter.camera_pose(states[-1,:9])
        assert np.linalg.norm(final[:3,3]-goal[:3,3])<.008
        assert result['maximum_acceleration_ratio']<=1.00001
        assert result['minimum_self_clearance_lower_bound_m']>=.005
    finally:process.terminate();process.communicate(timeout=3.)


def test_husky_native_declared_last_second_all_actuator_brake_has_original_rate_bounds(tmp_path):
    folder=ROOT/'Robot_Husky_UR5/ocs2';binary=native_worker(folder)
    if not binary.is_file():pytest.skip('build genuine native terminal-hold SLQ')
    sp=importlib.util.spec_from_file_location('all_brake_husky_adapter',folder/'adapter.py')
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);adapter=module.WheeledBackendAdapter()
    model=current_camera_collision_fixture(adapter,tmp_path)
    q=np.array(adapter.spec['initial_arm']);state=np.r_[np.zeros(3),q,[.03,-.06],np.full(6,.02)]
    base=np.eye(4);base[2,3]=adapter.nominal_height;camera=adapter.measured_camera_pose(base,q)
    request=dict(request_id=1,state_stamp_s=14.2,horizon_s=1.,state=state.tolist(),rate_state_semantics='last_applied_velocity_command',
                 T_world_base=base.tolist(),frame='world_optical',absolute_times_s=[10.1,15.2],camera_poses=[camera.tolist(),camera.tolist()],
                 terminal_hold=dict(mode='constant_original_endpoint_after_end',original_end_s=10.1,T_world_original_endpoint=camera.tolist(),chassis_brake_applied=True,all_actuators_brake_applied=True,all_actuator_brake_offset_s=4.))
    process=subprocess.Popen([str(binary),str(folder/'backend.json'),str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        process.stdin.write(json.dumps(request)+'\n');process.stdin.flush();result=json.loads(process.stdout.readline())
        assert result['ok'] is True,result
        assert result['chassis_brake_mode']=='all_actuator_constrained_deceleration'
        states=np.array(result['states']);times=np.array(result['times_s'])
        np.testing.assert_allclose(states[0],state,atol=1e-12)
        assert result['maximum_acceleration_ratio']<=1.00001
        assert result['maximum_velocity_ratio']<=1.00001
        assert np.max(np.abs(states[-1,9:]))<1e-5
        tail=states[times>=14.7]
        assert np.max(np.ptp(tail[:,:9],axis=0))<1e-5
    finally:process.terminate();process.communicate(timeout=3.)


def scene_module():
    sp=importlib.util.spec_from_file_location('single_wheeled_ground',ROOT/'scripts/ocs2_wheeled_scene.py')
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);return module


def ground_fixture():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/Ground').AddTranslateOp().Set((0,0,.2))
    visual=UsdGeom.Mesh.Define(stage,'/World/Ground/geom');UsdPhysics.CollisionAPI.Apply(visual.GetPrim())
    plane=UsdGeom.Plane.Define(stage,'/World/Ground/collisionPlane');plane.CreateAxisAttr('Z');UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    material=UsdShade.Material.Define(stage,'/World/GroundMaterial');physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim());physics.CreateStaticFrictionAttr(.8);physics.CreateDynamicFrictionAttr(.6);physics.CreateRestitutionAttr(0.)
    return stage,material


def test_real_usd_duplicate_floor_rejected_then_fixed_without_changing_robot():
    stage,material=ground_fixture()
    from pxr import UsdGeom,UsdPhysics,Sdf
    robot=UsdGeom.Cube.Define(stage,'/World/Robot/chassis');UsdPhysics.CollisionAPI.Apply(robot.GetPrim())
    before=Sdf.Layer.CreateAnonymous();Sdf.CreatePrimInLayer(before,'/World');Sdf.CopySpec(stage.GetRootLayer(),'/World/Robot',before,'/World/Robot')
    with pytest.raises(ValueError,match='exactly'):scene_module().audit_single_ground(stage,'/World/Ground',material,.2)
    report=scene_module().repair_single_ground(stage,'/World/Ground',material,.2)
    assert report['active_ground_colliders']==['/World/Ground/collisionPlane']
    assert report['static_friction']==pytest.approx(.8) and report['dynamic_friction']==pytest.approx(.6)
    after=Sdf.Layer.CreateAnonymous();Sdf.CreatePrimInLayer(after,'/World');Sdf.CopySpec(stage.GetRootLayer(),'/World/Robot',after,'/World/Robot')
    assert before.ExportToString()==after.ExportToString()


def test_real_usd_unknown_extra_ground_collider_remains_fail_closed():
    stage,material=ground_fixture()
    from pxr import UsdGeom,UsdPhysics
    extra=UsdGeom.Plane.Define(stage,'/World/Ground/rogue');UsdPhysics.CollisionAPI.Apply(extra.GetPrim())
    with pytest.raises(ValueError,match='exactly'):scene_module().repair_single_ground(stage,'/World/Ground',material,.2)


def test_real_usd_single_ground_rejects_wrong_height_or_material():
    stage,material=ground_fixture()
    from pxr import UsdShade
    scene_module().repair_single_ground(stage,'/World/Ground',material,.2)
    with pytest.raises(ValueError,match='height'):scene_module().audit_single_ground(stage,'/World/Ground',material,.3)
    other=UsdShade.Material.Define(stage,'/World/Other')
    with pytest.raises(ValueError,match='material'):scene_module().audit_single_ground(stage,'/World/Ground',other,.2)
