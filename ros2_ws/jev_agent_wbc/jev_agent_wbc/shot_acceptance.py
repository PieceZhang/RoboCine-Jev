"""User-aligned execution acceptance, separate from legacy reference fidelity.

This module evaluates saved/measured motion; it never generates a trajectory.
Visibility and reference fidelity are diagnostics, not acceptance gates.
"""
from copy import deepcopy

import numpy as np

from .camera_intent import POLICY, IntentSpec, visibility
from .cinematic_geometry import curve_quality
from .contracts import native


ACCEPTANCE_PROTOCOL = dict(
    version="shot_acceptance/v1",
    primary_checks=["safety", "direction", "natural_curve", "composition", "terminal_frame"],
    direction_reverse_limit=dict(orbit_rad=.003, translation_m=.004),
    shape=dict(total_turn_rad_exclusive=3.5, max_step_turn_rad_exclusive=.45,
               time_samples=121, equal_arc_samples=81, initial_join_exemption=False),
    composition=dict(in_frame_min=.98, source="calibrated_subject_projection"),
    terminal=dict(centroid_error=POLICY.terminal_centroid_tolerance,
                  log_area_error=POLICY.terminal_log_area_tolerance,
                  bbox_log_error=POLICY.terminal_bbox_log_tolerance,
                  view_angle_error_rad=POLICY.terminal_view_tolerance_rad,
                  silhouette_axis_error_rad=POLICY.terminal_silhouette_axis_tolerance_rad,
                  hold_s=POLICY.terminal_hold_s, speed_m_s=POLICY.terminal_speed_m_s),
    completion="directed_motion_and_stable_terminal_frame_in_band; no_exact_reference_progress_endpoint",
    diagnostics_only=["occlusion", "visible_coverage", "reference_process_offset",
                      "reference_process_view_and_scale", "reference_range_and_clock"],
    threshold_source="unchanged_existing_strict_shape_direction_frame_and_safety_checks",
    supported_scope="ordered_directed_motion_phases; adjacent same-motion phases merged; explicit transitions checked separately",
)


def _check(value, reason=None, **evidence):
    return dict(status="unverified" if value is None else "pass" if value else "fail",
                pass_check=None if value is None else bool(value), reason=reason, **evidence)


def occlusion_diagnostics(times, measurements):
    """Record opacity independently of frame truncation; missing values stay unknown.

    Durations use the same conservative endpoint-interval rule as legacy scoring.
    Partial occlusion has visibility < 1; severe intervals use the existing .85
    visibility threshold. Neither threshold is an acceptance condition.
    """
    times=np.asarray(times, float)
    values=[m.get("visibility") for m in measurements]
    known=np.array([v is not None and np.isfinite(v) for v in values])
    vis=np.array([float(v) if k else np.nan for v,k in zip(values,known)])
    def intervals(threshold):
        out=[]; start=None
        for i in range(len(times)-1):
            blocked=known[i] and known[i+1] and min(vis[i],vis[i+1]) < threshold
            if blocked and start is None: start=float(times[i])
            if start is not None and (not blocked or i==len(times)-2):
                end=float(times[i+1] if blocked else times[i])
                out.append(dict(start_s=start,end_s=end,duration_s=end-start)); start=None
        return out
    partial=intervals(1.); severe=intervals(POLICY.visibility_floor)
    known_intervals=known[:-1]&known[1:]
    dt=np.diff(times)
    return dict(affects_acceptance=False, source="supplied_visibility_measurements",
        known_samples=int(known.sum()),unknown_samples=int((~known).sum()),
        unknown_interval_s=float(dt[~known_intervals].sum()),
        visibility_min=float(vis[known].min()) if known.any() else None,
        visibility_mean=float(vis[known].mean()) if known.any() else None,
        max_occluded_fraction=float(1.-vis[known].min()) if known.any() else None,
        partial_occlusion_intervals=partial,
        partial_occlusion_total_s=sum(x["duration_s"] for x in partial),
        severe_visibility_threshold=POLICY.visibility_floor,
        severe_occlusion_intervals=severe,
        severe_occlusion_total_s=sum(x["duration_s"] for x in severe),
        longest_severe_occlusion_s=max((x["duration_s"] for x in severe),default=0.))


def evaluate_shot(original, model, times, positions, rotations, measurements, progress_times,
                  *, safety_pass, execution_ok=True, intrinsics=None, motion_end_s=None,
                  legacy_intent=None, safety_source="execution_constraint_checks"):
    """Five independent gates plus diagnostics; no reference endpoint snapping.

    `safety_pass` must come from the execution backend / independent physical
    audit, never be inferred from camera poses. Unknown evidence is unverified.
    `motion_end_s` excludes the terminal hold from the historical shape sampler.
    """
    times=np.asarray(times,float); positions=np.asarray(positions,float)
    rotations=np.asarray(rotations,float); progress=np.asarray(progress_times,float)
    if (len(times)<2 or positions.shape!=(len(times),3) or rotations.shape!=(len(times),3,3)
        or len(measurements)!=len(times) or progress.shape!=times.shape
        or not all(np.isfinite(x).all() for x in (times,positions,rotations,progress))
        or np.any(np.diff(times)<=0)):
        raise ValueError("finite ordered aligned execution samples are required")
    spec=IntentSpec(original,model,intrinsics)
    legacy=(spec.score(times,positions,rotations,measurements,progress)
            if legacy_intent is None else deepcopy(legacy_intent))
    # A bad process frame must fail composition, not invalidate a separately
    # recovered terminal hold. The legacy helper rejects any invalid process
    # projection, so give it only the original terminal observation window.
    window=times>=times[-1]-POLICY.terminal_hold_s-1e-7
    framing=spec.frame_and_trend(times[window],positions[window],rotations[window],progress[window])
    terminal=framing["terminal"]
    end=min(float(original.duration if motion_end_s is None else motion_end_s),float(times[-1]))
    class Executed:
        activation_s=float(times[0])
        duration=end
        def sample(self,t):
            return np.array([np.interp(t,times,positions[:,j]) for j in range(3)]),np.eye(3),np.zeros(6)
        def progress_time(self,t):
            return float(np.interp(t,times,progress))
    executed=Executed(); executed.original=original
    supported=bool(spec.segments) and all(s['supported'] for s in spec.segments)
    quality=curve_quality(executed) if supported and end>times[0] else None
    # The old geometric clock must stay ordered, but its final value need not
    # reach original.duration when the chosen terminal frame is already valid.
    ordered=bool(np.all(np.diff(progress)>=-1e-7) and progress[0]<=1e-6)
    motion=False
    phase_evidence=[]
    if supported:
        for segment in spec.segments:
            ids=np.flatnonzero((progress>=segment['start_s']-1e-7)&(progress<=segment['end_s']+1e-7))
            moved=False
            if len(ids)>=2:
                feature=spec.features(segment,positions[ids],rotations[ids])
                sign=np.sign(segment['end_value']-segment['start_value'])
                moved=bool(np.isfinite(feature).all() and sign*(feature[-1]-feature[0])>1e-7)
            phase_evidence.append(dict(motion=segment['motion'],start_s=segment['start_s'],
                end_s=segment['end_s'],samples=len(ids),motion_present=moved))
        motion=all(p['motion_present'] for p in phase_evidence)
    # Project the unoccluded subject with the existing silhouette-grid test.
    # Occluders and missing visibility do not masquerade as out-of-frame content.
    clear_scene=dict(valid=True,observation_stamp=0.,centers=[],radii=[],velocities=[])
    projected=[visibility(p,r,clear_scene,float(t),model,spec.intrinsics)
               for p,r,t in zip(positions,rotations,times)]
    contained=[m["in_frame"] for m in projected]
    composition_known=all(v is not None and np.isfinite(v) for v in contained)
    minimum=float(min(contained)) if composition_known else None
    checks=dict(
        safety=_check(safety_pass,source=safety_source),
        direction=_check(None if quality is None else quality["direction_pass"] and ordered and motion,
            reason=None if supported else "unsupported_motion_segment",
            motion_present=motion,progress_order_valid=ordered,
            phases=phase_evidence,
            reverse_motion=None if quality is None else quality["reverse_motion"],
            reverse_unit=None if quality is None else quality["reverse_unit"]),
        natural_curve=_check(None if quality is None else quality["strict_full_curve_pass"],
            measurements=quality),
        composition=_check(None if not composition_known else minimum>=.98,
            in_frame_min=minimum,required_in_frame_min=.98,
            source="unoccluded_calibrated_subject_projection_not_RGB"),
        terminal_frame=_check(terminal.get("pass_check"),measurements=terminal))
    from .combo import is_combo
    if is_combo(original):
        from .floating_camera import camera_motion_limits
        limits=camera_motion_limits(list(zip(positions,rotations)),times)
        checks['transition_continuity']=_check(limits['camera_limits_pass'],
            measurements=limits,source='measured_full_combo_speed_acceleration_angular_speed_including_joins')
    failed=[name for name,value in checks.items() if value["status"]=="fail"]
    unknown=[name for name,value in checks.items() if value["status"]=="unverified"]
    if not execution_ok: failed.append("execution_error")
    protocol=deepcopy(ACCEPTANCE_PROTOCOL)
    if is_combo(original):
        protocol['version']='shot_acceptance/v2_combo'
        protocol['primary_checks'].append('transition_continuity')
        protocol['shape']['scope']='each_main_action_and_explicit_transition; full_combo_sum_is_diagnostic'
    return native(dict(protocol=protocol,checks=checks,
        status="fail" if failed else "unverified" if unknown else "pass",
        accepted=bool(not failed and not unknown),failed_checks=failed,unverified_checks=unknown,
        motion_complete=bool(checks["direction"]["pass_check"] and terminal.get("pass_check")),
        diagnostics=dict(occlusion=occlusion_diagnostics(times,measurements),
            reference_fidelity=dict(affects_acceptance=False,legacy_intent=legacy)),
        evidence_scope="calibrated_geometry_and_supplied_safety_evidence; not_RGB_or_hardware"))


def attach_acceptance(summary, acceptance):
    """Publish the new headline while retaining historical scalar semantics."""
    summary["shot_acceptance"]=acceptance
    summary["legacy_reference_intent_success"]=summary.get("intent_success")
    summary["legacy_within_budget_success"]=summary.get("within_budget_success")
    summary["intent_success"]=acceptance["accepted"]
    # Timing is reported separately; it cannot resurrect reference-tracking gates.
    summary["within_budget_success"]=bool(acceptance["accepted"] and summary.get("time_overrun_s",0.)<=POLICY.terminal_hold_s+1e-6)
    summary["executed_camera_intent_role"]="legacy_reference_fidelity_diagnostic_only"
    return summary
