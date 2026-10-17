#!/usr/bin/env python3
"""Verify stabilized Agent routing and live original-reference PhysX cameras."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.whole_body_qp import whole_body_kinematics


def read(path):return json.loads(path.read_text())


def audit(folder):
    rows=read(folder/'telemetry.json');frames=read(folder/'video/frames.json')
    summary=read(folder/'summary.json');scene=read(folder/'runtime_scene.json')
    declared=read(folder/'video/camera_output.json')
    reference=OriginalReference(folder/'original_task.json',scene['subject']['position'])
    expected=read(folder/'video/reference_rendering.json')
    assert summary['camera_output_source']==declared['source']=='stabilized'
    assert (folder/'video/camera').resolve()==(folder/'video/stabilized').resolve()
    assert expected['valid'] and expected['same_render_pass'] and not expected['physics_body']
    assert expected['protocol']=='original_reference_floating_camera_same_scene_time/v1'
    position_errors=[];orientation_errors=[];physical_errors=[];correction=[]
    for row in rows:
        consumed=row['info']['agent_camera_input']
        assert row['camera_output_source']==consumed['camera_output_source']=='stabilized'
        assert abs(consumed['stamp_s']-row['t'])<1e-8
        np.testing.assert_allclose(consumed['camera_position'],row['stabilized_camera_position'],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(row['camera_position'],row['stabilized_camera_position'],atol=1e-10,rtol=0.)
        consumed_rotation=Rotation.from_quat(consumed['camera_quaternion_xyzw'])
        selected_rotation=Rotation.from_quat(row['stabilized_camera_quaternion_xyzw'])
        assert (consumed_rotation*selected_rotation.inv()).magnitude()<1e-9
        fk,_,_=whole_body_kinematics(np.asarray(row['state']),row['base_z'],np.asarray(row['base_rotation']))
        physical_errors.append(float(np.linalg.norm(fk-row['actual_camera_position'])))
        correction.append(float(np.linalg.norm(np.array(row['camera_position'])-row['actual_camera_position'])))
    by_time={round(row['t'],8):row for row in rows}
    for frame in frames:
        p,r,_=reference.sample(frame['t'])
        position_errors.append(float(np.linalg.norm(p-frame['reference_camera_position'])))
        orientation_errors.append(float((Rotation.from_matrix(r)*Rotation.from_quat(frame['reference_camera_quaternion_xyzw']).inv()).magnitude()))
        assert frame['reference_scene_time_s']==frame['t']
        assert frame['reference_camera_source']=='original_reference_floating_camera'
        assert abs(frame['physical_time_before_capture']-frame['physical_time_after_capture'])<1e-8
        row=by_time[round(frame['t'],8)]
        np.testing.assert_allclose(frame['camera_position'],row['stabilized_camera_position'],atol=1e-10,rtol=0.)
        assert all((folder/'video'/view/f"{frame['frame']:05d}.jpg").is_file()
                   for view in ('overview','reference','stabilized'))
    assert max(position_errors,default=0.)<1e-10 and max(orientation_errors,default=0.)<1e-9
    assert max(physical_errors,default=0.)<1e-4
    api=[json.loads(line) for line in (folder/'jev_api.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    decision_camera_checks=0
    for event in events:
        if event['event'] not in ('strategy_requested','search_configuration_requested','selection_requested'):
            continue
        feedback=event['evidence'].get('actual_robot_feedback') or {}
        if not feedback:continue
        assert feedback['camera_output_source']=='stabilized'
        row=by_time[round(feedback['stamp_s'],8)]
        np.testing.assert_allclose(feedback['camera_position'],row['stabilized_camera_position'],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(feedback['state'],row['state'],atol=1e-10,rtol=0.)
        decision_camera_checks+=1
    feedback_sources=[]
    for entry in api:
        state=entry['request']['state'].split('\nRUNTIME EVIDENCE:\n',1)
        if len(state)==2:
            evidence=json.loads(state[1]);feedback=evidence.get('actual_robot_feedback') or {}
            if 'camera_output_source' in feedback:
                feedback_sources.append(feedback['camera_output_source'])
    assert all(source=='stabilized' for source in feedback_sources)
    return dict(case=summary['case'],folder=str(folder.resolve()),passed=True,
        telemetry_samples=len(rows),rgb_frames=len(frames),agent_source='stabilized',
        agent_pose_matches_stabilized=True,agent_decision_requests_verified=decision_camera_checks,
        model_payloads_with_explicit_camera_source=len(feedback_sources),
        original_reference_position_error_max_m=max(position_errors,default=0.),
        original_reference_rotation_error_max_rad=max(orientation_errors,default=0.),
        actual_robot_fk_error_max_m=max(physical_errors,default=0.),
        virtual_camera_correction_max_m=max(correction,default=0.),
        same_scene_time_and_render_pass=True,reference_camera_has_physical_body=False,
        acceptance=summary['shot_acceptance']['status'],completed=summary['completed'])


def audit_film(video):
    """Decode a frame per chapter and compare each pane with its source JPEG."""
    import cv2
    from PIL import Image
    manifest=read(video.with_suffix('.json'))
    assert manifest['reference_rgb_displayed'] and not manifest['physical_camera_rgb_displayed']
    capture=cv2.VideoCapture(str(video));checks=[]
    try:
        for run,timeline in zip(manifest['runs'],manifest['timeline']):
            views=run['physx_rgb_views']
            assert [view['stream'] for view in views]==['overview','reference','stabilized']
            assert views[2]['title']=='增稳相机（Agent输入）' and views[2]['agent_input']
            folder=Path(run['folder']);frames=read(folder/'video/frames.json')
            tick=len(timeline['frame_times_s'])//2
            source_time=timeline['frame_times_s'][tick]
            encoded_frame=round(run['output_start_s']*manifest['fps'])+tick
            capture.set(cv2.CAP_PROP_POS_FRAMES,encoded_frame)
            ok,bgr=capture.read();assert ok
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            source_index=max(0,np.searchsorted([f['t'] for f in frames],source_time+1e-9,side='right')-1)
            errors=[]
            for column,view in enumerate(views):
                path=folder/'video'/view['stream']/f'{source_index:05d}.jpg'
                with Image.open(path) as original:
                    original=original.convert('RGB');original.thumbnail((624,300))
                    expected=Image.new('RGB',(624,300),'#0b111c')
                    expected.paste(original,((624-original.width)//2,(300-original.height)//2))
                x=16+640*column
                # Omit the bottom intent ribbon and the surrounding UI.
                actual=rgb[203:423,x+120:x+500].astype(float)
                desired=np.asarray(expected)[20:240,120:500].astype(float)
                error=float(np.mean(abs(actual-desired)));assert error<10.,(folder,view,error)
                errors.append(dict(stream=view['stream'],rgb_mean_absolute_error=error))
            checks.append(dict(folder=str(folder),output_frame=encoded_frame,
                source_time_s=source_time,rgb_acquisition_s=frames[source_index]['t'],panes=errors,passed=True))
    finally:
        capture.release()
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video',type=Path,help='also verify decoded film panes against recorded source RGB')
    args=parser.parse_args()
    results=[audit(folder) for folder in args.runs]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(recordings=results,film=audit_film(args.video) if args.video else None)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for result in results:print(result['case'],result['passed'],result['rgb_frames'])


if __name__=='__main__':main()
