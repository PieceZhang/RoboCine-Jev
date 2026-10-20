"""Provisional observed-camera phase checkpoints, never final task acceptance."""
import numpy as np
from scipy.spatial.transform import Rotation
from .reference import OriginalReference
from .shot_acceptance import evaluate_shot
from .camera_intent import POLICY


def evaluate_phase(original,phase,history,subject,intrinsics,max_sample_gap_s=.2):
    from .continuous_combo import action_task
    start,end=phase['start_s'],phase['end_s']
    selected=[r for r in history if start-1e-8<=r['t']<=end+POLICY.terminal_hold_s+1e-8]
    if len(selected)<3:return None
    times=np.array([r['t']-start for r in selected])
    # A sparse/stalled observation stream cannot prove a completed phase.
    if times[0]>.05 or times[-1]<end-start+POLICY.terminal_hold_s-.05 or np.max(np.diff(times))>max_sample_gap_s+1e-9:
        return None
    ref=OriginalReference(action_task(original.task,phase),original.subject)
    return evaluate_shot(ref,subject,times,[r['camera_position'] for r in selected],
        [Rotation.from_quat(r['camera_quaternion_xyzw']).as_matrix() for r in selected],
        [r['visual'] for r in selected],times,safety_pass=True,execution_ok=True,
        intrinsics=intrinsics,motion_end_s=ref.duration,
        safety_source='provisional Agent checkpoint; current certificates required, independent physical latch remains authoritative')
