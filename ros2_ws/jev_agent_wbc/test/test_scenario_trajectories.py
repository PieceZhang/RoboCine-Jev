"""The authored fixtures must exercise the moving arm and the mobile base."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jev_agent_wbc.kc_svsdf import CollisionGeometry
from jev_agent_wbc.nominal import nominal_rollout
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.scenarios import NAMES, RuntimeScene, generate
from jev_agent_wbc.whole_body_qp import whole_body_kinematics


ROOT = Path(__file__).resolve().parents[1]/'scenarios'


def test_checked_in_scenarios_match_generator(tmp_path):
    generate(tmp_path)
    for path in tmp_path.rglob('*.json'):
        assert json.loads(path.read_text()) == json.loads((ROOT/path.relative_to(tmp_path)).read_text())


@pytest.mark.parametrize('name', NAMES)
def test_large_shots_challenge_arm_during_whole_body_motion(name):
    scene = RuntimeScene(ROOT/name/'runtime_scene.json')
    fixture = scene.document
    reference = OriginalReference(ROOT/name/'task.json', fixture['subject']['position'])
    times, states, failure = nominal_rollout(reference, fixture['initial_state'], fixture['base_z'])
    assert not failure
    assert times[-1] == reference.duration
    assert np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum() > .4
    assert np.sum(np.ptp(states[:, 3:], axis=0) > .2) >= 2
    delta = reference.sample(reference.duration)[0]-reference.sample(0.)[0]
    assert np.linalg.norm(delta[:2]) >= .6
    if name.startswith('pedestal'):
        assert abs(delta[2]) >= .14
        assert np.sign(delta[2]) == (1 if name.endswith('up') else -1)
    geometry = CollisionGeometry()
    clearances = []
    for t, state in zip(times, states):
        p, r, _ = whole_body_kinematics(state, fixture['base_z'])
        target_p, target_r, _ = reference.sample(t)
        assert np.linalg.norm(p-target_p) < .025
        assert Rotation.from_matrix(r@target_r.T).magnitude() < .025
        obstacle, = scene.obstacles(t)
        clearances.append(geometry.distances(state, [obstacle['center']], fixture['base_z'])[0]-obstacle['radius'])
    clearances = np.array(clearances)
    # Exclude the base (0) AND camera (7): the actual moving arm must intersect.
    assert clearances[:, 1:7].min() < -.05
    assert clearances[:, 0].min() > .25
    assert clearances[[0, -1]].min() > .25
    crossing = clearances[:, 1:7].min(axis=1) < 0
    assert times[crossing].min() > 3.
    assert times[crossing].max() < 6.
