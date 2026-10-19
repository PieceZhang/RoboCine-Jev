"""Retire an obsolete camera detour at a measured, stopped action boundary.

Startup geometry is a proposal, not a permanent obstacle assessment. A shorter
path needs a new whole-body witness and current observed clearance. No future
scene motion enters this search; execution still certifies every command.
"""
import time
import numpy as np

from .intent_paths import IntentPath, positions_at, poses_at
from .phase_program import build
from .reference import smoothstep
from .transfer_geometry import prepare_arrival, capture_reference
from .kinematic_feasibility import KinematicSearch, summary
from .whole_body_qp import Limits
from .dynamic_collision import DynamicCollision
from .self_collision import SelfCollision
from .cinematic_geometry import curve_quality
from .floating_camera import camera_motion_limits
from .jev import choice


def path_length(path):
    return float(np.linalg.norm(np.diff(positions_at(path,np.linspace(0.,path.duration,401)),axis=0),axis=1).sum())


def without_detour(reference):
    """Remove only the avoidance bow; keep timing and the selected endpoint."""
    original=reference.original
    timings=reference.program.get('phase_timing',{})
    if not timings:return None
    candidate=build(original,timings,reference.model,reference.intrinsics,
                    {p['name']:'original' for p in original.task['shot_phase']})
    geometry=dict(candidate.geometry)
    times=np.asarray(geometry['times']);progress=np.asarray(geometry['progress_times'])
    # Refinement may have placed the terminal frame inside its allowed band.
    # Do not undo that endpoint adjustment when deleting the transverse bow.
    offset=reference.sample(reference.duration)[0]-candidate.sample(candidate.duration)[0]
    geometry['positions']=(np.asarray(geometry['positions'])+
        smoothstep(progress/candidate.duration)[:,None]*offset).tolist()
    candidate=IntentPath(original,reference.previous,0.,dict(candidate.program,arrival_detour_refresh=True),
                         reference.model,reference.intrinsics,geometry=geometry)
    return candidate if path_length(candidate)+.02<path_length(reference) else None


def screen(candidate,state,velocity,scene,z=0.,now=0.):
    """Fresh evidence for shortening, separate from mandatory execution safety."""
    age=now-float(scene.get('observation_stamp',-np.inf))
    report=dict(observation_stamp_s=scene.get('observation_stamp'),observation_age_s=age,
                scope='full FK and frozen observed occupancy; bounded motion lease, not future safety')
    if age<-.001 or age>.25 or not len(scene.get('surfel_centers',[])):
        return None,dict(report,eligible=False,reason='missing_or_stale_observation')
    quality=curve_quality(candidate)
    ts=np.linspace(0.,candidate.duration,401);pp,rr=poses_at(candidate,ts)
    limits=camera_motion_limits(list(zip(pp,rr)),ts)
    if not quality['direction_pass'] or not quality['strict_full_curve_pass'] or not limits['camera_limits_pass']:
        return None,dict(report,eligible=False,reason='camera_geometry',quality=quality,camera_limits=limits)
    proof=KinematicSearch(state,velocity,0.,z,Limits(),measured_control=velocity).lift(candidate,candidate.duration)
    report['kinematics']=summary(proof)
    if not proof['feasible'] or proof.get('min_joint_margin_rad',0.)<.08:
        return None,dict(report,eligible=False,reason='whole_body_allocation')
    states=np.asarray([row['state'] for row in proof['trace']])
    collision=DynamicCollision(scene,now)
    frozen=float((collision.geometry.distances_sequence(states,collision.centers[None],z)-collision.radii[None,:,None]).min())
    self_clear=float(SelfCollision().distances_sequence(states).min())
    report.update(frozen_observed_clearance_m=frozen,sampled_self_clearance_m=self_clear)
    if frozen<.10 or self_clear<.008:
        return None,dict(report,eligible=False,reason='observed_corridor_occupied')
    local=dict(scene,observation_stamp=scene['observation_stamp']-now)
    lease=KinematicSearch(state,velocity,0.,z,Limits(),measured_control=velocity,scene=local).lift(candidate,min(2.5,candidate.duration))
    report.update(observed_motion_lease=summary(lease),quality=quality,camera_limits=limits)
    if not lease['feasible'] or lease.get('min_body_clearance_m',0.)<.08:
        return None,dict(report,eligible=False,reason='observed_motion_conflict')
    return proof,dict(report,eligible=True,reason='shorter_path_has_current_evidence')


def propose(reference,state,velocity,scene,z,now,choose):
    """Worker-only search/API. The caller must recheck the returned proposal."""
    started=time.monotonic()
    candidate=without_detour(reference)
    record=dict(request_time_s=now,old_length_m=path_length(reference),selected=False)
    if candidate is None:return None,dict(record,reason='already_simple')
    candidate,allocation=prepare_arrival(candidate,state,z)
    candidate=capture_reference(candidate,state,velocity,z,keep_seed=True)
    proof,evidence=screen(candidate,state,velocity,scene,z,now)
    record.update(new_length_m=path_length(candidate),evidence=evidence,body_allocation=allocation)
    if proof is None:return None,dict(record,reason=evidence['reason'])
    # Use the verified body allocation as a preference, not an exact reset.
    geometry=dict(candidate.geometry,joint_seed=proof['trace'],joint_seed_preference_only=True)
    candidate=IntentPath(candidate.original,candidate.previous,0.,candidate.program,
                         candidate.model,candidate.intrinsics,geometry=geometry)
    record['numerical_screen_wall_s']=time.monotonic()-started
    api_started=time.monotonic()
    answer=choose(dict(stage='refresh_obsolete_arrival_detour',time_s=now,
        old_program=reference.program,observed_tracks=scene.get('tracks',[]),
        observed_objects=scene.get('object_observations',[]),**record,
        contract='Startup avoidance expires as evidence changes. Prefer the shorter original spatial action when its current full-body geometry, camera limits and motion lease pass. Preserve the endpoint, timing and all physical checks. No full-shot dynamic safety promise.'),
        {'path':choice('The shorter path passed current numerical gates. Remove the old bow unless current evidence justifies retaining it.',
             {'shorter':'Use the verified shorter spatial action.', 'retain':'Keep the previously selected bow.'})})['path']['choice']
    if answer not in ('shorter','retain'):raise ValueError('invalid arrival path choice')
    record['api_wall_s']=time.monotonic()-api_started
    return (candidate if answer=='shorter' else None),dict(record,selected=answer=='shorter',reason='official_jev_'+answer)
