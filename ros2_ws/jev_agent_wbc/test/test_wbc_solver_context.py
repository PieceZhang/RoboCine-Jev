"""A fresh old-task reply must never acquire the next task's authority."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from jev_agent_wbc.representation import Pose
from jev_agent_wbc.wbc_protocol import CameraTrajectory, TimedTask


SPEC = importlib.util.spec_from_file_location('g1_solver_context_test',
    Path(__file__).resolve().parents[1]/'Robot_G1/solver_context.py')
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)


def task():
    return TimedTask('camera', 1., 'physical-observation',
        CameraTrajectory(1., (0., 2.), (Pose((0., 0., 0.)), Pose((.1, 0., 0.)))),
        .05, .15)


def test_context_uses_original_protocol_camera_fingerprint():
    value = task()
    result = CONTEXT.task_context(value.to_dict())
    assert result['camera_path_id'] == value.trajectory.id
    assert CONTEXT.reply_status(result, deepcopy(result), result) == 'same_task'


def test_pre_task_context_cannot_be_applied_after_task_admission():
    current = CONTEXT.task_context(task().to_dict())
    assert CONTEXT.reply_status(None, None, None) == 'same_task'
    assert CONTEXT.reply_status(None, None, current) == 'superseded_task'


@pytest.mark.parametrize('changed', ['id', 'path', 'tolerance', 'observation', 'issued_at'])
def test_full_task_binding_rejects_still_fresh_previous_context(changed):
    original = task().to_dict()
    previous = CONTEXT.task_context(original)
    next_task = deepcopy(original)
    if changed == 'id':
        next_task['task_id'] = 'next'
    elif changed == 'path':
        next_task['camera_trajectory']['samples'][1]['pose']['translation_m'][0] += .1
    elif changed == 'tolerance':
        next_task['position_tolerance_m'] = .04
    elif changed == 'observation':
        next_task['observation_id'] = 'next-physical-observation'
    else:
        next_task['issued_at_s'] += .01
    current = CONTEXT.task_context(next_task)
    assert CONTEXT.reply_status(previous, previous, current) == 'superseded_task'
    assert CONTEXT.reply_status(current, previous, current) == 'reply_context_mismatch'


def test_snapshot_has_no_caller_mutation_alias():
    context = CONTEXT.task_context(task().to_dict())
    copied = CONTEXT.snapshot(context)
    context['task_id'] = 'mutated'
    assert copied['task_id'] == 'camera'


@pytest.mark.parametrize('bad', [False, {}, {'schema': 'other'},
    dict(schema=CONTEXT.SCHEMA, task_id='camera', camera_path_id='a'*64,
         task_sha256='G'*64),
    dict(schema=CONTEXT.SCHEMA, task_id='camera', camera_path_id='a'*64,
         task_sha256='b'*64, extra=True)])
def test_malformed_context_never_passes(bad):
    context = CONTEXT.task_context(task().to_dict())
    with pytest.raises(ValueError):
        CONTEXT.snapshot(bad)
    assert CONTEXT.reply_status(context, bad, context) == 'invalid_context'


def test_echo_is_required_for_an_active_task():
    context = CONTEXT.task_context(task().to_dict())
    assert CONTEXT.reply_status(context, None, context) == 'reply_context_mismatch'


def test_context_does_not_retain_the_mutable_task():
    source = task().to_dict()
    context = CONTEXT.task_context(source)
    source['camera_trajectory']['samples'].clear()
    assert context == CONTEXT.task_context(task().to_dict())
