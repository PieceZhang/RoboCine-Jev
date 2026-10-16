#!/usr/bin/env python3
"""Compare executed camera geometry and independently audit every CPU interval."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_state_search import audit
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.cinematic_geometry import curve_quality,terminal_band
from jev_agent_wbc.candidates import restore_path,path_hash


def inspect(folder, independent=False):
    summary=json.loads((folder/'summary.json').read_text())
    rows=json.loads((folder/'telemetry.json').read_text())
    runtime=json.loads((folder/'runtime_scene.json').read_text())
    original=OriginalReference(folder/'original_task.json',runtime['subject']['position'])
    times=np.array([r['t'] for r in rows]);points=np.array([r['camera_position'] for r in rows])
    class Executed:
        duration=min(original.duration,times[-1])
        activation_s=0.
        def sample(self,t):
            return np.array([np.interp(t,times,points[:,j]) for j in range(3)]),np.eye(3),np.zeros(6)
    executed=Executed();executed.original=original
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    cards={c['id']:c for e in events for c in e.get('cards',[])}
    installed=[]
    for e in events:
        if e['event']!='plan_installed':continue
        card=cards[e['candidate_id']];path=restore_path(original,card['trajectory'])
        assert path_hash(path)==e['candidate_id']
        installed.append(dict(time_s=e['time_s'],program=path.program,quality=curve_quality(path),
                              endpoint_offset_m=(path.sample(path.duration)[0]-original.sample(original.duration)[0]).tolist(),
                              predicted_terminal_pass=card['visual']['intent']['terminal_frame']['pass_check']))
    intent=summary['executed_camera_intent']
    api=[json.loads(line) for line in (folder/'jev_api.jsonl').read_text().splitlines()]
    result=dict(case=runtime['name'],folder=str(folder.resolve()),completed=summary['completed'],
        safety_pass=summary['safety_pass'],installed_plans=summary['installed_plans'],
        intent_success=summary['intent_success'],executed_curve=curve_quality(executed),
        terminal_frame=intent['terminal_frame'],motion_trend=intent['motion_trend'],
        visible_coverage=intent['effective_coverage'],occlusion_s=intent['longest_occlusion_s'],
        min_clearance_m=summary['min_truth_clearance_m'],installed=installed,
        api_calls=len(api),api_errors=sum('error' in r for r in api),
        actual_models=sorted({r['response']['model'] for r in api if r.get('response',{}).get('model')}),
        terminal_region=terminal_band(original,runtime['subject']))
    if independent:result['independent_audit']=audit(folder)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=[]
    for name in ['orbit','movein','truck_left','truck_right','pedestal_up','pedestal_down']:
        folder=args.runs/name/(name+'_jev');before=args.baseline/name/(name+'_jev')
        for filename in ['original_task.json','runtime_scene.json']:
            assert json.loads((folder/filename).read_text())==json.loads((before/filename).read_text())
        entry=dict(case=name,before=inspect(before),after=inspect(folder,True))
        result.append(entry)
        (args.output/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(name,'reverse',entry['before']['executed_curve']['reverse_motion'],
              '->',entry['after']['executed_curve']['reverse_motion'],
              'safe',entry['after']['independent_audit']['independent_safety_pass'],flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,2,figsize=(11,17))
    for i,entry in enumerate(result):
        for label,color in [('before','#999999'),('after','#008e7b')]:
            rows=json.loads((Path(entry[label]['folder'])/'telemetry.json').read_text())
            xyz=np.array([r['camera_position'] for r in rows]);t=[r['t'] for r in rows]
            axes[i,0].plot(xyz[:,0],xyz[:,1],color=color,label=label)
            axes[i,1].plot(t,xyz[:,2],color=color,label=label)
        axes[i,0].set_title(entry['case']+' camera XY');axes[i,1].set_title('camera height vs scene time')
        for ax in axes[i]:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'executed_paths.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
