"""Legged backend ownership, real Pink execution, and exact mixed-frame cameras."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


go2 = module('go2_native_backend', ROOT/'Robot_Go2/wbc_backend.py')
g1 = module('g1_pink_backend', ROOT/'Robot_G1/wbc_backend.py')
servo = module('legged_velocity_servo', ROOT/'Robot_G1/velocity_servo.py')
coordinator = module('go2_camera_coordinator', ROOT/'Robot_Go2/camera_coordinator.py')


def pose(position, rpy):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = position
    return result


def test_go2_mixed_frame_roundtrip_preserves_tilted_camera_and_full_extrinsics():
    rng = np.random.default_rng(20261020)
    for _ in range(100):
        link = pose(rng.normal(size=3), rng.uniform([-.6, -.6, -3.], [.6, .6, 3.]))
        camera = pose(rng.normal(size=3), rng.uniform(-2., 2., size=3))
        extrinsic = pose(rng.uniform(-.2, .2, size=3), rng.uniform(-2., 2., size=3))
        command = go2.camera_to_mixed(camera, link, extrinsic)
        reconstructed = go2.mixed_to_world_camera(command, link, extrinsic)
        np.testing.assert_allclose(reconstructed, camera, atol=1e-12)
        ee = camera@np.linalg.inv(extrinsic)
        np.testing.assert_allclose(command[:2], (np.linalg.inv(link)@ee)[:2, 3], atol=1e-12)
        assert command[2] == pytest.approx(ee[2, 3])


def test_rotating_xy_then_overwriting_world_z_is_not_the_inverse_when_tilted():
    link = pose([1., 2., .5], [.35, -.4, .2])
    command = np.r_[.5, .1, .8, 1., 0., 0., 0.]
    exact = go2.mixed_to_world_ee(command, link)
    naive = link[:3, 3]+link[:3, :3]@np.r_[command[:2], 0.]
    naive[2] = command[2]
    assert np.linalg.norm(exact[:3, 3]-naive) > .01
    np.testing.assert_allclose((np.linalg.inv(link)@exact)[:2, 3], command[:2], atol=1e-12)


@pytest.mark.parametrize('attitude', [[0., np.pi/2, 0.], [np.pi/2, 0., 0.]])
def test_singular_mixed_frame_is_rejected(attitude):
    link = pose([0., 0., .3], attitude)
    with pytest.raises(ValueError, match='singular'):
        go2.camera_to_mixed(np.eye(4), link, np.eye(4))
    with pytest.raises(ValueError, match='singular'):
        go2.mixed_to_world_ee([.5, 0., .5, 1., 0., 0., 0.], link)


def test_g1_height_ownership_has_no_policy_pink_joint_overlap():
    import yaml
    data = yaml.safe_load((ROOT/'Robot_G1/config/height_policy_io.yaml').read_text())
    ownership = g1.ownership(data['actions'][0]['joint_names'], data['articulations']['robot']['joint_names'])
    assert len(ownership['policy']) == 12 and len(ownership['pink']) == 15
    assert not set(ownership['policy']) & set(ownership['pink'])
    assert set(ownership['hold']) == {'waist_roll_joint', 'waist_pitch_joint'}
    assert sum(x['shape'][0] for x in data['observations']['policy']) == 80
    with pytest.raises(ValueError):
        g1.ownership(list(ownership['policy'])+['waist_yaw_joint'], data['articulations']['robot']['joint_names'])


@pytest.mark.parametrize('xyz,euler,violations', [
    ([.5, 0., .5], [0., 0., 0.], []),
    ([.4, -.35, .1], [-3.14/4, 0., 3.14/3], []),
    ([-.236, 0., .5], [0., 0., 0.], ['x_link0']),
    ([.71, .36, .81], [0., 0., 0.], ['x_link0', 'y_link0', 'z_world']),
    ([.5, 0., .5], [0., -1., 0.], ['pitch_link0']),
])
def test_go2_upstream_outer_domain_only_excludes_without_certifying_feasibility(xyz, euler, violations):
    quat = Rotation.from_euler('xyz', euler).as_quat()[[3, 0, 1, 2]]
    command = np.r_[xyz, quat]; original = command.copy()
    result = go2.check_upstream_domain(command)
    assert result['violations'] == violations
    assert result['supported'] is (False if violations else None)
    assert result['feasible'] is None and result['checkpoint_training_coverage'] == 'unknown'
    np.testing.assert_array_equal(command, original)


def test_go2_missing_domain_metadata_is_unknown_and_current_executed_adapter_is_preserved(tmp_path):
    import ast
    result = go2.check_upstream_domain([.5, 0., .5, 1., 0., 0., 0.], config_path=tmp_path/'missing.py')
    assert result['supported'] is False and result['feasible'] is None
    assert result['status'] == 'capability_metadata_unknown'
    executed = ROOT/'Robot_Go2/results/wbc_camera_coordinated_01/wbc_backend_source.py'
    if executed.is_file():
        def functions(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        before, after = functions(executed), functions(ROOT/'Robot_Go2/wbc_backend.py')
        for name in ('rigid', 'camera_to_mixed', 'mixed_to_world_ee', 'mixed_to_world_camera', 'set_camera_target'):
            assert before[name] == after[name]


def test_real_installed_pink_converges_without_changing_policy_or_base_coordinates():
    python = ROOT/'Robot_G1/.venv/bin/python'
    if not python.is_file():
        pytest.skip('isolated G1 runtime is not installed')
    script = r'''
import importlib.util,json,numpy as np,yaml
from pathlib import Path
from scipy.spatial.transform import Rotation
p=Path(__import__('sys').argv[1]);s=importlib.util.spec_from_file_location('actual_g1_backend',p/'wbc_backend.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
io=yaml.safe_load((p/'config/height_policy_io.yaml').read_text());names=io['articulations']['robot']['joint_names'];q=dict(zip(names,io['articulations']['robot']['default_joint_pos']))
camera=np.eye(4);camera[:3,3]=[.2,.04,.11];camera[:3,:3]=np.column_stack([[0,-1,0],[0,0,-1],[1,0,0]])
c=m.PinkUpperBody(io['actions'][0]['joint_names'],names,camera,posture_home=q.copy(),posture_cost=.03,damping_cost=.02)
base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.7]).as_matrix();base[:3,3]=[1.,2.,.72]
reference=c.predicted_camera(q,base);reference[:3,3]+=[.015,-.01,.01]
reference[:3,:3]=Rotation.from_rotvec([.01,-.015,.02]).as_matrix()@reference[:3,:3]
locked={n:q[n] for n in c.owners['policy']+c.owners['hold']};previous={n:0. for n in c.owners['pink']}
for _ in range(250):
 v=c.velocity(q,base,reference,.02)
 assert set(v)==set(c.owners['pink'])
 for name,speed in v.items():
  assert abs(speed)<=.5+1e-6 and abs(speed-previous[name])<=.02+1e-6
  q[name]+=speed*.02
 previous=v
assert all(q[n]==value for n,value in locked.items())
actual=c.predicted_camera(q,base)
ep=np.linalg.norm(actual[:3,3]-reference[:3,3]);er=Rotation.from_matrix(actual[:3,:3]@reference[:3,:3].T).magnitude()
assert ep<.002 and er<.01,(ep,er)
assert c.model.nq==29 and c.model.nv==29
print(json.dumps(dict(position_error_m=ep,orientation_error_rad=er,actual_pink=True,free_base_optimized=False)))
'''
    result = subprocess.run([str(python), '-c', script, str(ROOT/'Robot_G1')], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['actual_pink']


def test_pink_timeout_does_not_leave_an_unbounded_hold_only_success(monkeypatch):
    import time
    worker = object.__new__(g1.PinkProcess)
    worker.max_age_s = .06
    worker.pending = dict(request=dict(seq=1, stamp_s=0.), sent_wall_s=time.monotonic()-1.)
    worker.failure_count = 0; worker.needs_reset = False
    terminated = []
    worker.process = type('Worker', (), {'terminate': lambda self: terminated.append(True)})()
    monkeypatch.setattr(worker, '_receive', lambda wait: None)
    result = worker.poll(.02)
    assert not result['ready'] and 'deadline_exceeded' in result['status']
    assert terminated and worker.failure_count == 3 and worker.needs_reset


@pytest.mark.parametrize('held,reply_seq,reply_stamp,current_stamp', [
    (True, 4, 1., 1.02), (False, 3, 1., 1.02), (False, 4, .5, 1.02), (False, 4, 1., 1.2),
])
def test_pink_late_or_mismatched_response_cannot_resume_old_velocity(
        monkeypatch, held, reply_seq, reply_stamp, current_stamp):
    import time
    worker = object.__new__(g1.PinkProcess)
    worker.max_age_s = .06; worker.failure_count = 0; worker.needs_reset = False
    worker.pending = dict(request=dict(seq=4, stamp_s=1.), sent_wall_s=time.monotonic(),
                          held_since_submit=held, measured_camera=np.eye(4))
    monkeypatch.setattr(worker, '_receive', lambda wait: dict(ok=True, seq=reply_seq, stamp_s=reply_stamp,
        velocity={'waist_yaw_joint': .2}, predicted_camera=np.eye(4).tolist()))
    result = worker.poll(current_stamp)
    assert not result['ready'] and 'velocity' not in result
    assert worker.pending is None and worker.needs_reset and worker.failure_count == 1


def test_measured_velocity_servo_removes_deadband_bias_reverses_and_brakes_without_windup():
    controller = servo.VelocityServo()
    measured = np.zeros(3); previous = np.zeros(3)
    endpoints = []
    for target in ([.10, -.08, .12], [-.10, .08, -.12], [0., 0., 0.]):
        for _ in range(600):
            applied = controller.update(target, measured, .02)
            assert np.all(np.abs(applied) <= controller.limits+1e-10)
            assert np.all(np.abs(applied-previous) <= controller.slew*.02+1e-10)
            response = .7*np.sign(applied)*np.maximum(np.abs(applied)-.04, 0.)
            measured += .1*(response-measured)
            previous = applied
        endpoints.append(measured.copy())
    np.testing.assert_allclose(endpoints, [[.10, -.08, .12], [-.10, .08, -.12], [0., 0., 0.]], atol=.002)
    assert np.max(np.abs(controller.integral)) < .1


def test_support_guard_back_calculation_does_not_accumulate_hidden_velocity_request():
    controller = servo.VelocityServo(limits=(.2, .2, .2))
    for _ in range(300):
        controller.update([.2, 0., 0.], [0., 0., 0.], .02)
        controller.applied([0., 0., 0.])
    assert np.max(np.abs(controller.integral)) < .03
    assert np.max(np.abs(controller.update([0., 0., 0.], [0., 0., 0.], .02))) < .011


def test_stop_anchor_uses_physical_body_axes_including_nonzero_world_heading():
    current = pose([1., 2., .7], [0., 0., np.pi/2])
    anchor = pose([1.01, 1.98, .7], [0., 0., np.pi/2+.03])
    np.testing.assert_allclose(servo.stop_pose_error(anchor, current), [-.02, -.01, .03], atol=1e-12)


def test_go2_camera_priority_guard_allows_supported_motion_and_limits_conflicting_static_task():
    guard = coordinator.CameraCoordinator(np.eye(4), backend=go2)
    link = pose([0., 0., .35], [0., 0., 0.]); body = pose([0., 0., .28], [0., 0., 0.])
    def static(t):
        return pose([.5, 0., .5], [0., 0., 0.])
    def moving(t):
        return pose([.5+.1*t, 0., .5], [0., 0., 0.])
    conflict = guard.command(0., static, link, body, [.2, 0., 0.], [0., 0., 0.])
    assert conflict['camera_supported'] and not conflict['base_velocity_feasible']
    assert .01 < conflict['base_command'][0] < .15
    assert conflict['camera_reference_changed'] is False
    support = guard.command(0., moving, link, body, [.1, 0., 0.], [.1, 0., 0.])
    assert support['camera_supported'] and support['base_velocity_feasible']
    np.testing.assert_allclose(support['base_command'], [.1, 0., 0.], atol=1e-8)


def test_go2_camera_priority_horizon_brakes_before_leaving_static_world_workspace():
    guard = coordinator.CameraCoordinator(np.eye(4), backend=go2)
    link = pose([0., 0., .35], [0., 0., 0.]); body = pose([0., 0., .28], [0., 0., 0.])
    target = pose([.5, 0., .5], [0., 0., 0.]); original = target.copy()
    measured = np.zeros(3); dt = .02; positions = []
    for step in range(200):
        result = guard.command(step*dt, lambda t: target, link, body, [.2, 0., 0.], measured)
        assert result['camera_supported']
        measured += dt/.15*(result['base_command']-measured)
        link[:2, 3] += measured[:2]*dt; body[:2, 3] += measured[:2]*dt
        positions.append(go2.camera_to_mixed(target, link, np.eye(4))[0])
    assert min(positions) > .4 and link[0, 3] > .02
    assert abs(measured[0]) < .005
    np.testing.assert_array_equal(target, original)


def test_compiled_policy_domain_matches_pinned_file_checks_without_runtime_parsing():
    domain=go2.PolicyDomain()
    for xyz,euler in [([.5,0.,.5],[0.,0.,0.]),([-.2,0.,.5],[0.,0.,0.]),([.5,0.,.5],[0.,-.9,0.])]:
        command=np.r_[xyz,Rotation.from_euler('xyz',euler).as_quat()[[3,0,1,2]]]
        compiled,direct=domain.check(command),go2.check_upstream_domain(command)
        for key in ['status','supported','feasible','violations','sources_sha256']:
            assert compiled[key]==direct[key]


def test_actual_fcl_worker_distinguishes_original_sphere_clearance_and_overlap(tmp_path):
    import select
    python=ROOT/'Robot_TIAGo/wbc/.venv/bin/python'
    if not python.is_file():pytest.skip('isolated native FCL worker interpreter unavailable')
    collision=module('legged_collision_test',ROOT/'Robot_G1/legged_collision.py')
    metadata=tmp_path/'shapes.json'
    metadata.write_text(json.dumps(dict(shapes=[dict(name=n,body=n,kind='sphere',radius=.1,center_body=[0.,0.,0.])
        for n in ['a','b']],allowed_pairs=[])))
    worker=collision.DistanceProcess(python,ROOT/'Robot_G1/legged_collision.py',metadata,tmp_path/'stderr.log')
    try:
        for stamp,separation,expected in [(0.,.25,'pass'),(.02,.15,'fail')]:
            assert worker.submit(stamp,{'a':np.eye(4),'b':pose([separation,0.,0.],[0.,0.,0.])})
            assert select.select([worker.process.stdout],[],[],1.)[0]
            result=worker.poll(stamp+.02)
            assert result['status']==expected
            assert result['minimum_clearance_lower_bound_m']==pytest.approx(separation-.2,abs=1e-6)
    finally:worker.close()


def test_go2_motor_caps_come_from_same_pinned_model_and_cover_all_18_joints():
    limits=module('piper_actuator_limits',ROOT/'Robot_Go2/actuator_limits.py')
    result=limits.load_motor_limits(ROOT/'Robot_Go2/vendor/LeggedManip_Lab/mujoco/robots/go2_piper/go2piper.xml')
    caps=result['effort_limits_nm']
    assert len(caps)==18 and caps['FL_hip_joint']==23.7 and caps['RR_calf_joint']==45.43
    assert [caps[f'joint{i}'] for i in range(1,7)]==[20.,20.,15.,7.,5.,5.]


@pytest.mark.parametrize('delay_s,gain',[(.2,.8),(.4,1.),(.6,1.2)])
def test_low_gain_g1_servo_handles_delayed_velocity_response_reversal_and_stop(delay_s,gain):
    controller=servo.VelocityServo(**servo.g1_profile('low_gain'))
    dt=.02;history=[np.zeros(3) for _ in range(round(delay_s/dt))];measured=np.zeros(3);end=[]
    for desired in [[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]]:
        for _ in range(1200):
            command=controller.update(desired,measured,dt)
            history.append(command);delayed=history.pop(0)
            measured+=dt/.2*(gain*delayed-measured)
            assert np.isfinite(measured).all() and np.max(np.abs(measured))<.3
        end.append(measured.copy())
    np.testing.assert_allclose(end,[[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]],atol=.002)
