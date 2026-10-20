#!/usr/bin/env python3
"""Read-only arm PD algebra and scalar discrete-pole sensitivity.

No dynamics rollout, physical execution, inferred mass matrix or acceptance gate
is produced here. Recorded end-state algebra is not the earlier issued torque.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    robot=Path(__file__).resolve().parents[1]
    source=robot/'upstream/WBC-AGILE/agile/rl_env/assets/robots/unitree_g1.py'
    framework=robot/'upstream/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py'
    io_path=robot/'config/height_policy_io.yaml'
    assignment=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='G1_29DOF' for t in n.targets))
    actuators=next(k.value for k in assignment.value.keywords if k.arg=='actuators')
    arms=next(v for k,v in zip(actuators.keys,actuators.values) if ast.literal_eval(k)=='arms')
    native={k.arg:ast.literal_eval(k.value) for k in arms.keywords
        if k.arg in ('stiffness','damping','armature')}
    def resolve(kind,name):
        return float(next(v for p,v in native[kind].items() if re.fullmatch(p,name)))
    rows=[json.loads(x) for x in (args.result/'measurements.jsonl').open()]
    audit=json.loads((args.result/'actuator_audit.json').read_text())
    names=audit['native_actuator_joint_names']['arms'];indices=[audit['joint_names'].index(n) for n in names]
    kp=np.array([resolve('stiffness',n) for n in names]);armature=np.array([resolve('armature',n) for n in names])
    kd=np.array(audit['native_pd_damping']['arms'][0],float)
    np.testing.assert_allclose(audit['native_pd_stiffness']['arms'][0],kp,rtol=0,atol=1e-7)
    np.testing.assert_allclose(kd,[5*resolve('damping',n) for n in names],rtol=0,atol=1e-6)
    io=yaml.safe_load(io_path.read_text())['articulations']['robot']
    np.testing.assert_allclose(armature,[io['default_joint_armature'][io['joint_names'].index(n)] for n in names],rtol=0,atol=1e-8)
    q=np.asarray([r['joint_positions'] for r in rows])[:,indices]
    ref=np.asarray([r['applied_joint_position_target'] for r in rows])[:,indices]
    qd=np.asarray([r['measured_joint_velocity'] for r in rows])[:,indices]
    vref=np.asarray([[r.get('pink_upper_velocity',{}).get(n,0.) for n in names] for r in rows])
    gravity=np.asarray([[r.get('upper_gravity_feedforward',{}).get(n,0.) for n in names] for r in rows])
    caps=np.asarray(audit['effort_limits'])[indices]
    dt=.005;profiles=[]
    for scale in (1.,2.,4.):
        stiffness=scale*kp;algebra=stiffness*(ref-q)+kd*(vref-qd)+gravity
        threshold=(stiffness*dt*dt+2*kd*dt)/4
        poles=[]
        for inertia in (.001,.002,.003,.01,.02,.03,.06,.12):
            joint=[]
            for n,k,d in zip(names,stiffness,kd):
                # Semi-implicit Euler, one scalar fixed target, torque at tick start.
                A=np.array([[1-dt*dt*k/inertia,dt*(1-dt*d/inertia)],[-dt*k/inertia,1-dt*d/inertia]])
                ev=np.linalg.eigvals(A)
                joint.append(dict(name=n,spectral_radius=float(max(abs(ev))),
                    poles_real_imag=[[float(z.real),float(z.imag)] for z in ev],
                    stable=bool(max(abs(ev))<1)))
            poles.append(dict(assumed_scalar_inertia_kg_m2=inertia,joints=joint))
        profiles.append(dict(arms_stiffness_scale=scale,
            joint_stiffness_Nm_rad=dict(zip(names,stiffness.tolist())),
            necessary_scalar_inertia_threshold_kg_m2=dict(zip(names,threshold.tolist())),
            maximum_necessary_scalar_inertia_threshold_kg_m2=float(max(threshold)),
            minimum_source_armature_minus_scalar_threshold_kg_m2=float(min(armature-threshold)),
            all_arms_source_armature_exceeds_scalar_threshold=bool(np.all(armature>threshold)),
            end_state_algebraic_PD_plus_g_over_cap=float(np.max(np.abs(algebra)/caps)),
            end_state_algebraic_clipped_components=int(np.sum(np.abs(algebra)>caps)),
            scalar_pole_sensitivity=poles))
    result=dict(schema='g1_arm_stiffness_algebra/v1',physical_execution=False,source_result=str(args.result),
        rows=len(rows),physics_dt_s=dt,source_armature_kg_m2=dict(zip(names,armature.tolist())),
        actual_physx_armature_readback_available=False,
        same_state_actual_damping_Nm_s_rad=dict(zip(names,kd.tolist())),profiles=profiles,
        selected_candidate_arms_stiffness_scale=2.,
        rationale='Gentler first physical comparison: greater scalar stability margin than x4, reduced reference lag without altering source caps or learned targets.',
        interpretation=[
            'The pinned G1_29DOF configuration supplies armature0.03; the different old delayed-motor configuration supplies0.02 and is not this execution source.',
            'Pinned IsaacLab _process_actuators_cfg writes actuator.armature to PhysX for explicit and implicit actuators; future Kit startup must read the actual PhysX values back.',
            'Scalar necessary stability inequality Kp*dt^2+2*Kd*dt<4*I assumes a constant uncoupled inertia and semi-implicit integration. It is not a coupled floating-base/contact stability certificate.',
            'Waist stiffness/damping are unchanged; no full-upper-body stability guarantee is inferred from armature alone.',
            'End-state PD+g algebra freezes measured references and states; it is not the issued earlier5ms torque, a future transient bound or a stopping/collision certificate.',
            'Low-inertia unstable counterexamples are retained in the pole sensitivity; actual physical behavior, original caps, FCL and full stop gates remain authoritative.'],
        evidence_sha256={})
    for path in (Path(__file__),source,framework,io_path,args.result/'measurements.jsonl',args.result/'actuator_audit.json'):
        result['evidence_sha256'][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__),args.output/'executed_source.py')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:p[k] for k in ('arms_stiffness_scale','maximum_necessary_scalar_inertia_threshold_kg_m2',
        'minimum_source_armature_minus_scalar_threshold_kg_m2','end_state_algebraic_PD_plus_g_over_cap',
        'end_state_algebraic_clipped_components')} for p in profiles],indent=2))


if __name__=='__main__':main()
