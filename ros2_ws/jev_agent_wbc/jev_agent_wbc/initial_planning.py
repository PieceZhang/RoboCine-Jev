"""One Jev-directed initial plan before the physical shot clock starts.

Arbitrary measured initial camera error is a boundary condition, not an
instruction to retrace the authored start. The original task and deadline are
immutable. Physical sessions may resume successful preparation stages for the
same measured input; each rejected construction still needs a new Jev choice.
"""
from dataclasses import asdict
import time
import numpy as np

from .contracts import digest
from .candidates import Approval, path_hash, installation_guard
from .directed_search import construction_evidence
from .search_configuration import offers
from . import task_space_planning


def prepare(engine, state, scene, base_z):
    if engine.mode!='jev' or not engine.budget.single_candidate or engine.active is not engine.original:
        raise ValueError('initial preparation requires official Jev single-candidate mode and an unstarted shot')
    if engine.jev.decider.client.url!='https://api.typesafe.ai/v1/systemone':
        raise ValueError('initial single-candidate experiments require TypeSafe official API')
    started=time.monotonic()
    original=engine.original
    scene=dict(scene,subject_model=engine.subject_model,intrinsics=engine.intrinsics)
    resumable=getattr(engine,'resumable_preparation',False)
    key=digest(dict(task=original.sha256,state=np.asarray(state).tolist(),scene=scene,base_z=float(base_z)))
    if not resumable or getattr(engine,'preparation_checkpoint_key',None)!=key:
        engine.preparation_checkpoint={}
        engine.preparation_checkpoint_key=key
    checkpoint=engine.preparation_checkpoint
    def stage(name,compute):
        if not resumable or name not in checkpoint:checkpoint[name]=compute()
        else:engine._write('preparation_checkpoint_resumed',0.,stage=name,input_sha256=key)
        return checkpoint[name]
    observation=engine.geometry_backend.observe(state,0.,base_z=base_z)
    diagnostic=task_space_planning.diagnose(original,scene,state,np.zeros(8),0.,base_z,engine.planning_limits,
        budget=engine.budget,measured_control=np.zeros(8))
    context=dict(objective='visible_intent_completion',time_s=0.,duration_s=original.duration,
        phase=original.phase(0.),shot_phase=original.task['shot_phase'],intent=original.task['intent'],subject=original.task['subject'],
        reference_subject_position=original.subject.tolist(),difficulty=diagnostic['report'],
        robot_geometry=observation.summary(engine.geometry_backend.obstacle_relations(observation,scene)),
        construction=dict(construction_evidence(original,original,scene,0.,0.,
            mount_height_m=observation.to_dict()['world_from_mount']['translation_m'][2]),measured_initial_boundary=True))
    evidence=dict(context,stage='strategy')
    if getattr(engine,'preparation_feedback',None):
        context['previous_preparation_failures']=engine.preparation_feedback
        evidence=dict(context,stage='strategy')
    engine._write('strategy_requested',0.,evidence=evidence,startup=True)
    first=stage('strategy',lambda:engine.jev.decider.decide(evidence))
    engine._write('search_strategy',0.,**first)
    evidence=dict(context,stage='configuration',search_options=offers(first['strategy'],None))
    engine._write('search_configuration_requested',0.,evidence=evidence,startup=True)
    second=stage('configuration',lambda:engine.jev.decider.decide(evidence))
    engine._write('search_configuration',0.,**second)
    result=stage('candidate',lambda:task_space_planning.build(original,original,first['strategy'],scene,state,np.zeros(8),0.,base_z,engine.planning_limits,
        budget=engine.budget,measured_control=np.zeros(8),commanded_q=state[3:],
        intrinsics=engine.intrinsics,search_config=second['search_config'],construction_plan=second['construction_plan'],
        initial_boundary=True,representation_boxes=engine.geometry_backend.box_budget,
        geometry_preferences=engine.geometry_preferences))
    candidates=result.get('candidates',[])
    engine._write('candidate_set',0.,cards=[c.card() for c in candidates],evaluations=result.get('evaluations',[]),
        planner_wall_s=result['wall_s'],seed_wall_s=result.get('seed_wall_s'),generated_proposals=result.get('generated_proposals'),
        construction_plan=second['construction_plan'],startup=True)
    if result.get('error') or len(candidates)!=1 or len(result.get('evaluations',[]))!=1:
        # A geometric rejection invalidates the generating decisions too.
        # Retaining the family can force every retry into the same infeasible
        # construction. Transport failures above still retain completed work.
        checkpoint.clear()
        engine.preparation_feedback=[dict(reason=result.get('error'),
            rejected_construction=second['construction_plan'],evaluations=result.get('evaluations'))]
        raise RuntimeError('initial_single_candidate_rejected:'+(result.get('error') or 'invalid_count'))
    candidate=candidates[0]
    evidence=dict(context,stage='selection',candidates=[candidate.card()])
    engine._write('selection_requested',0.,evidence=evidence,startup=True)
    selected=stage('selection',lambda:engine.jev.decider.decide(evidence))
    if selected['candidate_id']!=candidate.id:
        checkpoint.pop('selection',None)
        raise RuntimeError('initial_choice_binding_mismatch')
    approval=Approval(candidate.id,digest(candidate.card()),path_hash(original),original.sha256,
        selected['selected_probability'],0)
    engine._write('candidate_approved',0.,approval=asdict(approval),decision=selected,startup=True)
    verified=task_space_planning.verify(candidate,approval,original,scene,state,np.zeros(8),0.,base_z,
        engine.planning_limits,budget=engine.budget,commanded_q=state[3:],measured_control=np.zeros(8),
        intrinsics=engine.intrinsics)
    error=verified.get('error') or installation_guard(verified,original,scene,state,0.,engine.budget,
        measured_control=np.zeros(8),previous=np.zeros(8),commanded_q=state[3:],base_z=base_z)
    if error:
        checkpoint.clear()
        raise RuntimeError('initial_installation_rejected:'+error)
    engine.active=candidate.path;engine.active_candidate=candidate
    engine.wbc.set_trajectory(engine.active)
    engine.wbc.kinematic_reference=verified['kinematics']
    engine.wbc.geometry_guidance=verified.get('geometry_guidance')
    engine.configuration_monitor.reset(candidate.id,verified['kinematics'])
    engine.last_install_s=engine.last_attempt_s=0.
    engine.validation_attempts=engine.feasible_plans=engine.installed_plans=1
    wall=time.monotonic()-started
    engine.pipeline_latencies.append(wall)
    engine.pipeline_outcomes.append(dict(outcome='installed',time_s=0.,wall_s=wall,startup=True))
    engine._write('plan_installed',0.,candidate_id=candidate.id,approval=asdict(approval),activation_s=0.,
        checks=verified['checks'],startup=True,end_to_end_wall_s=wall,
        measured_start='camera position and orientation; original terminal task unchanged')
    return dict(wall_s=wall,planner_wall_s=result['wall_s'],candidate_id=candidate.id)
