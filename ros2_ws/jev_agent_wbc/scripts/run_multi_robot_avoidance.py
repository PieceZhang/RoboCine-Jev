#!/usr/bin/env python3
"""Serial pace=1 native Isaac runs; retain every failure and original artifact."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PACKAGE=Path(__file__).resolve().parents[1]
PYTHON='/home/w84384651/miniconda3/envs/env_isaaclab/bin/python'


def command(robot,case,mode,output,probe_m=.3,probe_axis='radial',probe_candidate_index=None,probe_family='joint',probe_activation_s=2.,obstacle_height_offset_m=0.):
    common=['--output',str(output),'--avoidance-case',case,'--avoidance-mode',mode,
            '--avoidance-obstacle-height-offset-m',str(obstacle_height_offset_m)]
    if mode=='probe':common+=['--avoidance-probe-m',str(probe_m),'--avoidance-probe-axis',probe_axis,
                              '--avoidance-probe-activation-s',str(probe_activation_s)]
    if mode=='probe' and probe_candidate_index is not None:
        common+=['--avoidance-probe-candidate-index',str(probe_candidate_index),'--avoidance-probe-family',probe_family]
    if robot in ('husky','ridgeback'):
        directory='Robot_Husky_UR5' if robot=='husky' else 'Robot_Ridgeback_UR5'
        return ['bash',str(PACKAGE/directory/'run_ocs2.sh'),*common]
    if robot=='tiago':
        return ['bash',str(PACKAGE/'Robot_TIAGo/run_giskard.sh'),*common]
    if robot=='g1':
        return ['bash',str(PACKAGE/'Robot_G1/run_basic_wbc.sh'),*common]
    return ['bash',str(PACKAGE/'Robot_Go2/run_adaptive_wbc.sh'),
        '--cases','camera_shot','--move-seconds','8','--stop-seconds','5','--fine-collision',
        '--go2-response-profile','measured_backward_v1','--go2-native-posture','x58_all_v1',
        '--go2-prediction-profile','bounded_ramp_history_v2','--go2-world-z-feedback','bounded_integral_v1',
        '--go2-world-z-seed','fresh_actual_output_continuity_v2',
        '--go2-terminal-profile','endpoint_xy58_15_yaw0_v1','--go2-active-stand-brake','vxy_p05_v1',
        '--camera-excursion','diagnostic','--ground-material','source_multiply',
        '--parking-anchor-reset','--park-damping-scale','3',*common]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robots',nargs='+',choices=['husky','ridgeback','tiago','g1','go2'],
                        default=['husky','ridgeback','tiago','g1','go2'])
    parser.add_argument('--cases',nargs='+',choices=['orbit','movein'],default=['orbit','movein'])
    parser.add_argument('--mode',choices=['jev','nominal','probe'],default='jev')
    parser.add_argument('--max-minutes',type=float,default=180.)
    parser.add_argument('--probe-m',type=float,default=.3)
    parser.add_argument('--probe-axis',choices=['radial','lateral','height'],default='radial')
    parser.add_argument('--probe-candidate-index',type=int,choices=range(6))
    parser.add_argument('--probe-family',choices=['framing','view','joint'],default='joint')
    parser.add_argument('--probe-activation-s',type=float,default=2.)
    parser.add_argument('--obstacle-height-offsets',nargs='*',default=[],metavar='ROBOT=METRES',
                        help='explicit scene-only height micro-adjustments; radius, horizontal motion and clock unchanged')
    args=parser.parse_args()
    height_offsets={}
    for item in args.obstacle_height_offsets:
        robot,value=item.split('=',1);value=float(value)
        if robot not in ('husky','ridgeback','tiago','g1','go2') or not -.15<=value<=.15:
            parser.error('height offsets require a known robot and finite value within +/-0.15 m')
        height_offsets[robot]=value
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();results=[]
    for robot in args.robots:
        for case in args.cases:
            remaining=args.max_minutes*60-(time.monotonic()-started)
            if remaining<=0:break
            destination=args.output/f'{robot}_{case}'
            cmd=command(robot,case,args.mode,destination,args.probe_m,args.probe_axis,
                        args.probe_candidate_index,args.probe_family,args.probe_activation_s,height_offsets.get(robot,0.))
            before=time.monotonic()
            with (args.output/f'{robot}_{case}.log').open('w') as log:
                try:
                    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=process.wait(timeout=min(180.,remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code='process_timeout'
            row=dict(robot=robot,case=case,command=cmd,returncode=code,
                     wall_s=time.monotonic()-before,output=str(destination))
            for name,file in [('avoidance','avoidance/summary.json'),('native','acceptance.json' if robot in ('husky','ridgeback','tiago') else 'summary.json')]:
                p=destination/file
                if p.is_file():row[name]=json.loads(p.read_text())
            if row.get('avoidance'):
                with (args.output/f'{robot}_{case}_audit.log').open('w') as log:
                    try:
                        subprocess.run([PYTHON,str(PACKAGE/'scripts/audit_multi_robot_avoidance.py'),str(destination)],
                                       stdout=log,stderr=subprocess.STDOUT,timeout=30.)
                    except subprocess.TimeoutExpired:
                        row['audit_error']='independent audit timeout; not accepted'
                p=destination/'avoidance/independent_audit.json'
                if p.is_file():row['audit']=json.loads(p.read_text())
            results.append(row)
            (args.output/'matrix.json').write_text(json.dumps(dict(mode=args.mode,results=results,
                requested_robots=args.robots,requested_cases=args.cases,
                obstacle_height_offsets_m=height_offsets,
                elapsed_wall_s=time.monotonic()-started),indent=2)+'\n')
            print(json.dumps(dict(robot=robot,case=case,returncode=code,wall_s=row['wall_s'],
                avoidance=row.get('avoidance'))),flush=True)
    return 0 if len(results)==len(args.robots)*len(args.cases) and all(r.get('audit',{}).get('successful') for r in results) else 2


if __name__=='__main__':
    raise SystemExit(main())
