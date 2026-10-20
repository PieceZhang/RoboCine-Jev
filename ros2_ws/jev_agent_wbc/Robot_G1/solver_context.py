"""Bind an asynchronous solver reply to the complete admitted task.

Sequence numbers and both command leases remain the transport's responsibility.
This small boundary prevents a still-fresh reply for the previous task from
being applied after a new task has been admitted. It does not authorize motion.
The caller must validate the TimedTask before constructing this context.
"""
from copy import deepcopy
from hashlib import sha256
import json


SCHEMA = 'g1_solver_task_context/v1'
FIELDS = frozenset(('schema', 'task_id', 'camera_path_id', 'task_sha256'))


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def _identifier(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 256


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def task_context(task):
    """Snapshot a validated serialized task once, at admission.

    The full-task digest also binds tolerances, the observation and any future
    supported geometry request, even if the task ID or camera path is reused.
    This function is not a substitute for the public protocol's validation.
    """
    if (not isinstance(task, dict) or task.get('schema') != 'jev_wbc/v1'
            or task.get('kind') != 'task' or not _identifier(task.get('task_id'))
            or not isinstance(task.get('camera_trajectory'), dict)):
        raise ValueError('an admitted serialized TimedTask is required')
    return dict(schema=SCHEMA, task_id=task['task_id'],
                camera_path_id=_digest(task['camera_trajectory']),
                task_sha256=_digest(task))


def snapshot(context):
    """Return a detached context; None is only the pre-task context."""
    if context is None:
        return None
    if (not isinstance(context, dict) or set(context) != FIELDS
            or context['schema'] != SCHEMA or not _identifier(context['task_id'])
            or not _hash(context['camera_path_id']) or not _hash(context['task_sha256'])):
        raise ValueError('invalid solver task context')
    return deepcopy(context)


def reply_status(request_context, reply_context, current_context):
    """Classify context only; freshness/sequence/success must pass separately.

    A superseded task is an ordinary admission transition, not a solver fault.
    A missing or forged echo for a current task is a transport/context failure.
    No status from this function alone makes a command executable.
    """
    try:
        request, reply, current = map(snapshot,
                                      (request_context, reply_context, current_context))
    except (ValueError, TypeError):
        return 'invalid_context'
    if request != current:
        return 'superseded_task'
    if reply != request:
        return 'reply_context_mismatch'
    return 'same_task'
