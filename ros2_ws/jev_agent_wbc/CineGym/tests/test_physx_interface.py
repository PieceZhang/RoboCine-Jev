from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from CineGym import CineGymEnv, Episode, build_command
from CineGym.actions import validate_action
from CineGym.backends import execution_environment, OFFICIAL_API
from CineGym.bridge import action_packet, observation
from CineGym.episode import fingerprint
from CineGym.physx_env import PhysXEnv
from jev_agent_wbc.contracts import native
from jev_agent_wbc.realtime import CommandWatchdog, RuntimePolicy
from jev_agent_wbc.whole_body_qp import Limits

PACKAGE = Path(__file__).resolve().parents[2]


@pytest.fixture
def source():
    folder = PACKAGE/'scenarios/movein'
    task = json.loads((folder/'task.json').read_text())
    scene = json.loads((folder/'runtime_scene.json').read_text())
    episode = Episode(task, scene, dict(seed=0, family='movein', difficulty='demo', sha256=fingerprint(task, scene)))
    original = episode.reference
    p, r, _ = original.sample(0.)
    snapshot = dict(t=0., dt=.02, sequence=3, observation_wall=10.,
        state=np.array(scene['initial_state']), measured_control=np.zeros(8),
        base_z=.002, base_rotation=np.eye(3), measured_base_twist=np.zeros(6),
        camera=dict(position=p, rotation=r, source='execution'),
        scene=dict(observation_stamp=0., surfel_centers=np.array([[2., 2., .2]])),
        control_status=dict(runtime_fault=None, accepted_action_sequence=-1),
        evaluation=dict(truth_clearance_m=.5))
    return episode, original, snapshot


def test_common_action_limits_cover_velocity_acceleration_wheels_and_joints(source):
    episode, original, snapshot = source
    valid = np.zeros(8); valid[0] = .002
    assert np.array_equal(validate_action(valid, snapshot['state'], np.zeros(8), .02, Limits()), valid)
    for action in ([0.]*7, [float('nan')]*8, [1.]*8):
        with pytest.raises(ValueError):
            validate_action(action, snapshot['state'], np.zeros(8), .02, Limits())
    wheels = Limits(wheel_acceleration=.01)
    with pytest.raises(ValueError, match='limits'):
        validate_action(valid, snapshot['state'], np.zeros(8), .02, wheels)


def test_physical_packet_uses_exact_observation_not_current_time(source):
    _, original, snapshot = source
    packet = action_packet(np.zeros(8), snapshot, original, Limits(), dict(jev_calls=0, installed_plans=0))
    assert packet['sequence'] == 3 and packet['observation_wall'] == 10. and packet['t'] == 0.
    guard = CommandWatchdog(RuntimePolicy(), 10.)
    assert guard.accept(packet, .02, 10.02)
    assert not guard.accept(dict(packet, sequence=4), .03, 10.11)
    guard.command(.03, 10.11, .03)
    assert guard.fault['reason'] == 'command_timeout'
    assert not guard.accept(dict(packet, sequence=5, t=.03, observation_wall=10.12), .03, 10.12)


def test_operating_reserve_survives_wire_format_and_gates_external_action(source):
    _, original, snapshot = source
    limits = Limits(acceleration=Limits().acceleration*.1)
    wire = json.loads(json.dumps(native(observation(snapshot, original, limits))))
    received = Limits(**wire['action_limits'])
    assert np.array_equal(received.acceleration, limits.acceleration)
    action = np.zeros(8); action[0] = .002
    action_packet(action, snapshot, original, Limits(), {})
    with pytest.raises(ValueError, match='limits'):
        action_packet(action, snapshot, original, received, {})


def test_observation_has_measured_rotation_but_no_truth_or_future(source):
    _, original, snapshot = source
    obs = observation(snapshot, original)
    assert np.array_equal(obs['base_rotation'], snapshot['base_rotation'])
    assert obs['camera_source'] == 'execution'
    assert 'evaluation' not in obs and 'obstacles' not in obs and 'truth_clearance_m' not in obs
    assert obs['lidar']['observation_stamp'] == 0.


def test_backend_factory_does_not_initialize_isaac_for_cpu_or_on_construction(tmp_path):
    with CineGymEnv(backend='cpu') as cpu:
        assert cpu.backend == 'cpu'
    with CineGymEnv(backend='physx', output=tmp_path/'physical') as physical:
        assert isinstance(physical._env, PhysXEnv)
        assert physical.handle is None
        with pytest.raises(RuntimeError, match='reset'):
            physical.step(np.zeros(8))
    with pytest.raises(ValueError, match='backend'):
        CineGymEnv(backend='invalid')
    with pytest.raises(ValueError, match='dt'):
        CineGymEnv(backend='physx', dt=.1)


def test_ros_environment_recovers_xacro_without_sourcing_shell(tmp_path):
    ros = tmp_path/'ros'
    (ros/'local/lib/python3.10/dist-packages').mkdir(parents=True)
    (ros/'bin').mkdir()
    original = dict(CINEGYM_ROS_PREFIX=str(ros), PYTHONPATH='/existing/path', PATH='/usr/bin')
    result = execution_environment('physx', original)
    assert str(ros/'local/lib/python3.10/dist-packages') in result['PYTHONPATH'].split(':')
    assert '/existing/path' in result['PYTHONPATH'].split(':')
    assert result['PATH'].startswith(str(ros/'bin'))
    assert original['PYTHONPATH'] == '/existing/path'


def test_default_jev_is_official_and_explicit_override_is_respected(source, tmp_path):
    episode, _, _ = source
    case = episode.save(tmp_path/'case')
    command = build_command(case, tmp_path/'run')
    assert command[command.index('--api-url')+1] == OFFICIAL_API
    custom = build_command(case, tmp_path/'run2', api_url='http://localhost:9999/test')
    assert custom[custom.index('--api-url')+1] == 'http://localhost:9999/test'


def prepared_env(source, tmp_path):
    episode, original, snapshot = source
    env = PhysXEnv(output=tmp_path, terminate_on_fault=True)
    env.episode, env.reference = episode, original
    env.current = observation(snapshot, original)
    env.current['_evaluation'] = None
    env.done = False
    env.wall_start = time.monotonic()
    env.connection = SimpleNamespace(send_bytes=lambda payload: None)
    return env


def test_latched_fault_truncates_and_does_not_leak_evaluation(source, tmp_path):
    env = prepared_env(source, tmp_path)
    next_obs = deepcopy(env.current)
    next_obs.update(t=.04, sequence=4, _evaluation=dict(stamp_s=.02, truth_clearance_m=.1,
        min_clearance_m=.1, collision_samples=0, physical_contacts=0))
    next_obs['control_status'] = dict(runtime_fault=dict(reason='command_timeout'), accepted_action_sequence=-1)
    env._next = lambda sequence, timeout: next_obs
    obs, reward, terminated, truncated, info = env.step(np.zeros(8))
    assert truncated and not terminated and env.done
    assert '_evaluation' not in obs and 'truth_clearance_m' not in obs
    assert info['evaluation_delay_s'] == .02 and info['truth_clearance_m'] == .1
    assert not info['action_accepted']


def test_missing_stream_is_truncation_not_completion(source, tmp_path):
    env = prepared_env(source, tmp_path)
    def timeout(*args):
        raise TimeoutError('no measured observations')
    env._next = timeout
    _, _, terminated, truncated, info = env.step(np.zeros(8))
    assert truncated and not terminated and not info['completed']
    assert info['truth_clearance_m'] is None


def test_physical_contact_penalty_and_termination(source, tmp_path):
    env = prepared_env(source, tmp_path)
    next_obs = deepcopy(env.current)
    next_obs.update(t=.02, sequence=4, _evaluation=dict(stamp_s=0., truth_clearance_m=-.01,
        min_clearance_m=-.01, collision_samples=1, physical_contacts=1))
    env._next = lambda sequence, timeout: next_obs
    _, reward, terminated, truncated, info = env.step(np.zeros(8))
    assert terminated and not truncated and info['collision'] and reward <= -1.
