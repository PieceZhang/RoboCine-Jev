#!/usr/bin/env python3
"""Read-only observed-prefix planner diagnostic; never a physical acceptance."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from jev_agent_wbc.multi_robot_runtime import AvoidanceRuntime
from jev_agent_wbc.multi_robot_geometry import NativeKinematicModel
from jev_agent_wbc.representation import Observation
from jev_agent_wbc.reference import OriginalReference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--stamp', type=float, required=True)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--robot', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = AvoidanceRuntime.__new__(AvoidanceRuntime)
    runtime.package, runtime.robot, runtime.output = PACKAGE, a.robot, a.output
    runtime.client=None
    runtime.events = (a.output/'events.jsonl').open('w', buffering=1)
    document = json.loads((a.source/'runtime_scene.json').read_text())
    runtime.subject_model=document['subject']
    from jev_agent_wbc.multi_robot_audit import camera_intrinsics
    runtime.intrinsics=camera_intrinsics(json.loads((a.source.parent/'sensor_mounts.json').read_text())['camera'])
    runtime.original = OriginalReference(json.loads((a.source/'original_task.json').read_text()),
                                         document['subject']['position'])
    runtime.model = NativeKinematicModel(json.loads((a.source/'native_model.json').read_text()))
    runtime.scene_start = a.start
    maps = [json.loads(s) for s in (a.source/'observed_maps.jsonl').read_text().splitlines()]
    observations=[Observation.from_dict(json.loads(line)) for line in
                  (a.source/'observations.jsonl').read_text().splitlines()]
    observation=max((o for o in observations if o.stamp_s<=a.stamp),key=lambda o:o.stamp_s)
    snapshot = [m['snapshot'] for m in maps if m['consumed_s'] <= observation.stamp_s][-1]
    if not 0<=observation.stamp_s-snapshot['observation_stamp']<=.25:
        raise ValueError('No fresh causally available measured map at the selected acquisition')
    events=[json.loads(line) for line in (a.source/'events.jsonl').read_text().splitlines()]
    start=next(e['shot_start_s'] for e in events if e['event']=='scenario_bound')
    if abs(a.start-start)>1e-9:
        raise ValueError('Diagnostic must retain the recorded original shot clock')
    native_states=[json.loads(line) for line in (a.source/'native_states.jsonl').read_text().splitlines()]
    state=next(s for s in native_states if s['observation_id']==observation.id)
    if state['stamp_s']!=observation.stamp_s:
        raise ValueError('Native state must bind the exact public acquisition')
    model=runtime.model.recalibrated(state['body_poses'],state['joint_positions'],state.get('native_platform_state'))
    try:
        chosen, result = runtime._plan(snapshot, observation,model.document)
        (a.output/'result.json').write_text(json.dumps(dict(
            physical_execution=False, selected=chosen is not None, result=result,
            observation_stamp_s=observation.stamp_s,native_model_scope='acquisition-bound calibrated kinematic model',
            map_acquired_s=snapshot['observation_stamp'], map_consumed_before_s=a.stamp), indent=2)+'\n')
    finally:
        if runtime.client is not None:
            runtime.client.closed = True
        runtime.events.close()


if __name__ == '__main__':
    main()
