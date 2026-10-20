#!/usr/bin/env python3
"""Three measured terminal samples: genuine Pin Jacobian/body compensation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('exact_park_backend',a.result/'sources/g1_backend.py')
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    rows=[json.loads(x) for x in (a.result/'measurements.jsonl').open()];by_time={round(r['stamp_s'],8):r for r in rows}
    names=rows[0]['joint_names'];legs=[n for n in names if any(s in n for s in ('hip','knee','ankle'))]
    rig=json.loads((a.result/'sensor_mounts.json').read_text())
    solver=backend.PinkUpperBody(legs,names,rig['camera']['T_parent_optical'],a.result/'sources/g1_29dof.urdf')
    pin=solver.pin;pelvis_com=np.array([0.,0.,-.07603006064891815]);samples=[]
    for case in ('camera_shot','yaw_left','yaw_right'):
        block=[r for r in rows if r['case']==case];mid=block[-1]['stamp_s']-.24
        applied=min((r for r in block if r['pink_upper_velocity']),key=lambda r:abs(r['stamp_s']-mid))
        stamp=applied['pink_runtime_recovery']['stamp_s'];measured=by_time[round(stamp,8)]
        prev=by_time[round(stamp-.02,8)];nxt=by_time[round(stamp+.02,8)]
        q=dict(zip(names,measured['joint_positions']));cfg,owned=solver.configuration(q)
        B=np.asarray(measured['T_world_base']);C=np.asarray(measured['T_world_camera_optical'])
        J=pin.computeFrameJacobian(cfg.model,cfg.data,cfg.q,cfg.model.getFrameId('cine_camera_optical'),pin.LOCAL_WORLD_ALIGNED)
        world=np.zeros((6,6));world[:3,:3]=B[:3,:3];world[3:,3:]=B[:3,:3];J=world@J
        inv=np.linalg.pinv(J);projector=inv@J
        v=np.array([applied['pink_upper_velocity'][n] for n in owned]);row=projector@v;null=v-row
        # Net actor motion from actual adjacent measured poses, holding relative
        # camera fixed; excludes arm motion and does not substitute raw COM twist.
        before=np.asarray(prev['T_world_base']);after=np.asarray(nxt['T_world_base'])
        c_before=before@np.linalg.inv(B)@C;c_after=after@np.linalg.inv(B)@C
        body_fd=np.r_[(c_after[:3,3]-c_before[:3,3])/.04,
            Rotation.from_matrix(c_after[:3,:3]@c_before[:3,:3].T).as_rotvec()/.04]
        v_actor_fd=(after[:3,3]-before[:3,3])/.04
        xy=np.r_[v_actor_fd[:2],0.,0.,0.,0.]
        omega=np.array(measured['base_angular_velocity_world']);vcom=np.array(measured['base_linear_velocity_world'])
        com_world=B[:3,3]+B[:3,:3]@pelvis_com
        raw_twist=np.r_[vcom+np.cross(omega,C[:3,3]-com_world),omega]
        compensation=-inv@body_fd;xy_comp=-inv@xy;native_twist=J@v
        predicted=solver.predicted_camera(q,B)
        from pink.tasks import FrameTask
        frame=FrameTask('cine_camera_optical',position_cost=1.,orientation_cost=.3,lm_damping=.001,gain=.1)
        frame.set_target(pin.SE3(np.linalg.inv(B)@np.asarray(applied['T_world_reference_optical'])))
        error_J=frame.compute_jacobian(cfg);error_projector=np.linalg.pinv(error_J)@error_J
        first_park=next(r for r in block if (r.get('parking_transition') or {}).get('entered'))
        admission=by_time[round(first_park['stamp_s']-.02,8)]
        old=applied['pink_parking_posture']['targets']
        admission_q=dict(zip(names,admission['joint_positions']))
        preference=np.array([old[n] if n!='waist_yaw_joint' else 0. for n in owned])
        refreshed=np.array([admission_q[n] if n!='waist_yaw_joint' else 0. for n in owned])
        cost=np.array([.15 if n=='waist_yaw_joint' else .03 for n in owned])
        N=np.eye(len(owned))-projector
        gradients={}
        for tag,target in [('current_native_preference',preference),('first_PARK_measured',refreshed)]:
            gradient=cost**2*(cfg.q-target)
            projected=-N@gradient
            gradients[tag]=dict(target=target.tolist(),posture_error_norm_rad=float(np.linalg.norm(cfg.q-target)),
                weighted_negative_gradient_null_norm=float(np.linalg.norm(projected)),
                cosine_native_null_vs_descent=float(null@projected/max(np.linalg.norm(null)*np.linalg.norm(projected),1e-30)),
                native_null_directional_derivative=float(null@gradient))
        values=dict(case=case,applied_row_stamp_s=applied['stamp_s'],native_acquisition_stamp_s=stamp,
            matrix_source='GenuinePin reduced15 camera LOCAL_WORLD_ALIGNED Jacobian, rotated by measured pelvisSE3; no floating-base execution.',
            owned_names=owned,J_world_linear_angular=J.tolist(),singular_values=np.linalg.svd(J,compute_uv=False).tolist(),
            FK_position_error_m=float(np.linalg.norm(predicted[:3,3]-C[:3,3])),
            native_qdot=v.tolist(),native_norm_rad_s=float(np.linalg.norm(v)),
            native_rowspace_norm_rad_s=float(np.linalg.norm(row)),native_nullspace_norm_rad_s=float(np.linalg.norm(null)),
            native_nullspace_energy_fraction=float(null@null/max(v@v,1e-30)),
            genuine_FrameTask_projection_max_difference=float(np.max(np.abs(error_projector-projector))),
            current_native_posture_capture_stamp_s=applied['pink_parking_posture']['stamp_s'],
            first_PARK_admission_stamp_s=admission['stamp_s'],soft_posture_gradient_comparison=gradients,
            native_camera_world_twist=native_twist.tolist(),
            measured_actor_translation_fd_m_s=v_actor_fd.tolist(),
            measured_body_only_camera_twist_centered40ms=body_fd.tolist(),raw_COM_based_instant_camera_twist=raw_twist.tolist(),
            body_compensation_min_norm_qdot=compensation.tolist(),body_compensation_norm_rad_s=float(np.linalg.norm(compensation)),
            body_XY_only_compensation_min_norm_qdot=xy_comp.tolist(),body_XY_only_compensation_norm_rad_s=float(np.linalg.norm(xy_comp)),
            native_plus_body_twist=(native_twist+body_fd).tolist(),
            cosine_native_vs_body_compensation=float(v@compensation/max(np.linalg.norm(v)*np.linalg.norm(compensation),1e-30)),
            cosine_native_vs_XY_compensation=float(v@xy_comp/max(np.linalg.norm(v)*np.linalg.norm(xy_comp),1e-30)),
            raw_native_and_compensation_joint={n:dict(native_rad_s=float(v[i]),rowspace_rad_s=float(row[i]),
                nullspace_rad_s=float(null[i]),body_compensation_rad_s=float(compensation[i]),
                XY_compensation_rad_s=float(xy_comp[i])) for i,n in enumerate(owned)})
        samples.append(values)
    report=dict(schema='g1_park_body_compensation_three_samples/v1',physical_execution=False,samples=samples,
        scope='Three middle-of-final-halfsecond acquisition states; exact currentTask native commands, genuineJacobian and actualbody finitepose differences. Not a newphysicalpass or a causal plant identification.',
        interpretation=['Native command rowspace/nullspace are Euclideanjoint projections; SI tasktwist weighting is explicit via J, not a safety metric.',
            'Body-only40ms pose difference and instantaneous rawCOM twist are separately recorded; one is not substituted for the other.',
            'Minimum-norm compensation is a local kinematic possibility, not the commanded velocity or a guarantee that nativePARK XY feedback will work.',
            'Uncontrolled waistroll/pitch motion, nonlineargeometry, finitehistory and nativePosture/barriers are not removed from the actual controller.'],
        evidence_sha256={})
    for path in (Path(__file__),a.result/'measurements.jsonl',a.result/'sources/g1_backend.py',a.result/'sources/g1_29dof.urdf',a.result/'sensor_mounts.json'):
        report['evidence_sha256'][str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(Path(__file__),a.output/'executed_source.py')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in samples:
        print(json.dumps({k:r[k] for k in ('case','native_acquisition_stamp_s','native_norm_rad_s',
            'native_nullspace_energy_fraction','body_compensation_norm_rad_s','body_XY_only_compensation_norm_rad_s',
            'measured_actor_translation_fd_m_s','cosine_native_vs_body_compensation','cosine_native_vs_XY_compensation')},indent=2))


if __name__=='__main__':main()
