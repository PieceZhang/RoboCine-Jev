#!/usr/bin/env python3
"""Summarize immutable measured runs without promoting diagnostics to Agent success."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(source,audit_filename='independent_audit.json'):
    matrix_file=source/'matrix.json'
    matrix=json.loads(matrix_file.read_text())
    rows=[]
    hashes={str(matrix_file):hashlib.sha256(matrix_file.read_bytes()).hexdigest()}
    for entry in matrix['results']:
        directory=Path(entry['output'])
        if not directory.is_dir():
            directory=source/f"{entry['robot']}_{entry['case']}"
        summary_file=directory/'avoidance/summary.json'
        audit_file=directory/'avoidance'/audit_filename
        summary=json.loads(summary_file.read_text()) if summary_file.is_file() else {}
        audit=json.loads(audit_file.read_text()) if audit_file.is_file() else {}
        binding_file=directory/'avoidance/scenario_binding.json'
        scene_file=directory/'avoidance/runtime_scene.json'
        binding=json.loads(binding_file.read_text()) if binding_file.is_file() else {}
        scene=json.loads(scene_file.read_text()) if scene_file.is_file() else {}
        for file in (summary_file,audit_file,binding_file,scene_file):
            if file.is_file():hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        intent=audit.get('intent',{})
        relaxed=entry['robot'] in ('g1','go2')
        row=dict(robot=entry['robot'],case=entry['case'],mode=matrix['mode'],
            complete=summary.get('complete',False),returncode=entry['returncode'],
            obstacle_height_adjustment_m=binding.get('obstacle_height_adjustment_m'),
            additional_obstacle_height_offset_m=binding.get('additional_obstacle_height_offset_m',0.),
            obstacle_initial_heights_m=[o['keyframes'][0]['position'][2] for o in scene.get('obstacles',[])],
            scenario_duration_s=binding.get('duration_s'),
            tracking_pass=summary.get('native_tracking_pass',False),
            position_rmse_m=summary.get('position_rmse_m'),
            orientation_rmse_rad=summary.get('orientation_rmse_rad'),
            terminal_position_error_m=summary.get('terminal_position_error_m'),
            terminal_orientation_error_rad=summary.get('terminal_orientation_error_rad'),
            position_tolerance_m=.08 if relaxed else .03,
            orientation_tolerance_rad=.20 if relaxed else .08,
            measured_minimum_clearance_lower_bound_m=summary.get('measured_minimum_clearance_lower_bound_m'),
            observed_environment_pass=summary.get('all_environment_constraints_pass',False),
            measured_clearance_pass=summary.get('safety_pass',False),
            native_hard_constraints_pass=audit.get('native_hard_constraints_pass',False),
            native_hard_constraints_status=audit.get('native_hard_constraints_status','unclassified'),
            strict_stop_pass=audit.get('stop_pass',False),
            intent_complete=intent.get('intent_complete',False),
            intent_violation_kinds=[v['kind'] for v in intent.get('intent_violations',[])],
            timing_pass=audit.get('timing_pass',False),
            simulation_s=audit.get('simulation_s'),wall_s=audit.get('wall_s'),
            exact_agent_plan_installed=summary.get('exact_plan_installed',False),
            agent_trial_status=('accepted_physical_agent_trial' if audit.get('successful') else
                'installed_physical_acceptance_failed' if summary.get('exact_plan_installed') and audit else
                summary.get('agent_trial_status','unclassified')),
            physical_acceptance=audit.get('physical_acceptance',False),
            agent_successful=audit.get('successful',False),
            audit_file=str(audit_file) if audit_file.is_file() else None,
            output=str(directory))
        row['wall_to_simulation_ratio']=(row['wall_s']/row['simulation_s']
            if row['wall_s'] is not None and row['simulation_s'] else None)
        rows.append(row)
    return dict(schema='five_robot_measured_comparison/v1',source=str(source),
        mode=matrix['mode'],expected_cases=(len(matrix['requested_robots'])*len(matrix['requested_cases'])
            if 'requested_robots' in matrix else 10),recorded_cases=len(rows),
        complete_cases=sum(r['complete'] for r in rows),
        tracking_pass_count=sum(r['tracking_pass'] for r in rows),
        physical_acceptance_count=sum(r['physical_acceptance'] for r in rows),
        agent_success_count=sum(r['agent_successful'] for r in rows),
        scope='measured native physical diagnostics' if matrix['mode']!='jev' else 'official agent trials',
        input_sha256=hashes,results=rows)


def plot(report,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=report['results'];x=np.arange(len(rows));width=.35
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    settings=[
        ('position_rmse_m','terminal_position_error_m','position_tolerance_m',100.,'Position error (cm)'),
        ('orientation_rmse_rad','terminal_orientation_error_rad','orientation_tolerance_rad',180/np.pi,'Orientation error (degrees)')]
    for ax,(rms,terminal,limit,scale,label) in zip(axes,settings):
        values=lambda key:np.array([r[key] if r[key] is not None else np.nan for r in rows])*scale
        ax.bar(x-width/2,values(rms),width,label='RMS',color='#2374ab')
        ax.bar(x+width/2,values(terminal),width,label='Endpoint',color='#ed9b40')
        ax.plot(x,values(limit),'k_',markersize=26,markeredgewidth=2,label='Tracking threshold')
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(ncols=3,loc='upper left')
    axes[1].set_xticks(x,[f"{r['robot']}\n{r['case']}" for r in rows])
    for tick,row in zip(axes[1].get_xticklabels(),rows):
        tick.set_color('#137333' if row['tracking_pass'] else '#b3261e')
    title='Official Jev physical trials' if report['mode']=='jev' else 'Five-robot physical diagnostics'
    fig.suptitle(f"{title}: {report['tracking_pass_count']}/{len(rows)} tracking passes\n"
                 "Collision, native constraints, stop, intent and timing are audited separately",fontsize=14)
    fig.supxlabel('Label color: tracking pass / fail. Exact installation and full acceptance are separate checks.',fontsize=10)
    fig.savefig(destination/'tracking_comparison.png',dpi=160)
    fig.savefig(destination/'tracking_comparison.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-filename',default='independent_audit.json')
    args=parser.parse_args()
    report=summarize(args.source,args.audit_filename)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    rows=report['results']
    if rows:
        with (args.output/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        plot(report,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','results')},indent=2))


if __name__=='__main__':
    main()
