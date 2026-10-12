#!/usr/bin/env python3
"""Report all Isaac runs, including failures, and independent safety gates."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def series(rows,key):return np.array([float(r.get(key,0) or 0) for r in rows])

def statistics(folder):
    summary=json.loads((folder/'summary.json').read_text())
    rows=read_rows(folder/'telemetry.csv');tracking=[r for r in rows if r['phase']=='tracking']
    if not rows:return dict(completed=False,failure=summary.get('failure','no data')),rows
    t=series(rows,'time_s');u=np.column_stack([series(rows,'command_'+k) for k in ['v','omega']+[f'dq{i}' for i in range(1,7)]])
    scale=np.array([.5,1.2,1,1,1,1.2,1.2,1.5])
    steps=np.max(np.abs(np.diff(u,axis=0))/scale,axis=1) if len(u)>1 else np.zeros(1)
    plans=[r for r in rows if int(float(r.get('mpc_replanned',0)))];pt=series(plans,'mpc_update_s')
    end=[r for r in tracking if float(r['time_s'])>=float(tracking[-1]['time_s'])-1] if tracking else []
    rms=lambda rs,k:float(np.sqrt(np.mean(series(rs,k)**2))) if rs else None
    audit=summary.get('collision_audit',{})
    result=dict(completed=summary.get('completed',False),failure=summary.get('failure',''),observed_s=float(t[-1]),
        position_rmse_mm=1000*rms(tracking,'position_error') if tracking else None,
        orientation_rmse_deg=np.rad2deg(rms(tracking,'orientation_error')) if tracking else None,
        position_peak_mm=1000*float(np.max(series(tracking,'position_error'))) if tracking else None,
        end_position_rmse_mm=1000*rms(end,'position_error') if end else None,
        end_orientation_rmse_deg=np.rad2deg(rms(end,'orientation_error')) if end else None,
        minimum_clearance_mm=1000*audit.get('minimum_clearance_m',float('nan')),
        continuous_lower_mm=1000*audit.get('continuous_lower_m',float('nan')),
        collision_state_rate=audit.get('collision_sample_rate',float('nan')),
        unsafe_state_rate=audit.get('unsafe_sample_rate',float('nan')),
        mpc_mean_ms=1000*float(np.mean(pt)) if len(pt) else 0,
        mpc_p95_ms=1000*float(np.percentile(pt,95)) if len(pt) else 0,
        fast_qp_p95_ms=1000*float(np.percentile(series(rows,'solve_s'),95)),
        controller_p95_ms=1000*float(np.percentile(series(rows,'controller_s'),95)),
        max_normalized_control_step=float(np.max(steps)),
        max_command_violation=max([v for k,v in summary.items() if k.startswith('max_') and 'violation' in k and 'measured' not in k] or [0]),
        max_simultaneously_active_pairs=int(max(series(rows,'collision_active_pairs'))),
        kc_terms_max=int(max(series(rows,'kc_filtered_terms'))),
        planner_failures=int(max(series(rows,'avoidance_plan_failures'))),
        physical_audit_pass=summary.get('physics_audit',{}).get('pass_constraints',False),
        physical_maxima=summary.get('physics_audit',{}).get('maxima',{}),
        base_path_m=float(np.sum(np.hypot(np.diff(series(rows,'state_x')),np.diff(series(rows,'state_y'))))),
        joint_path_rad=float(np.sum(np.linalg.norm(np.diff(np.column_stack([series(rows,f'state_q{i}') for i in range(1,7)]),axis=0),axis=1))))
    result['gates']=dict(completed=result['completed'],continuous_clearance=bool(result['completed'] and audit.get('clearance_pass',False)),
        recovery=bool(result['completed'] and result['end_position_rmse_mm']<=20 and result['end_orientation_rmse_deg']<=np.rad2deg(.035)),
        command_constraints=result['max_command_violation']<=1e-6,
        bounded_control_steps=result['max_normalized_control_step']<=.05,
        physical_constraints=result['physical_audit_pass'])
    if 'combo' in folder.name:
        result['junctions']={}
        yaw=series(rows,'state_yaw')
        for junction in (22.,36.):
            indices=np.where((t[1:]>=junction-.2)&(t[1:]<=junction+.2))[0]
            if len(indices):
                result['junctions'][str(junction)]=dict(
                    max_normalized_control_step=float(np.max(steps[indices])),
                    max_measured_heading_step_deg=float(np.rad2deg(np.max(np.abs(np.diff(np.unwrap(yaw))[indices])))))
    with gzip.open(folder/'collision.csv.gz','rt') as f:
        collisions=list(csv.DictReader(f))
    body_gaps=np.column_stack([series(collisions,f'body{b}_clearance_m') for b in range(8)])
    result['base_and_arm_near_obstacles_fraction']=float(np.mean((body_gaps[:,0]<.1)&np.any(body_gaps[:,1:]<.1,axis=1)))
    result['max_simultaneous_bodies_near_obstacles']=int(np.max(np.sum(body_gaps<.1,axis=1)))
    return result,rows


def figures(name,runs,scene,out):
    colors=dict(none='#888888',per_link='#d88b20',kc='#1677b8')
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for mode,(stats,rows,folder) in runs.items():
        if not rows:continue
        t=series(rows,'time_s');color=colors[mode]
        axes[0,0].plot(t,series(rows,'position_error')*1000,label=mode,color=color)
        axes[0,1].plot(t,np.rad2deg(series(rows,'orientation_error')),label=mode,color=color)
        axes[1,0].plot(t,series(rows,'command_v'),label=mode,color=color)
        axes[1,1].plot(t,series(rows,'command_omega'),label=mode,color=color)
        with gzip.open(folder/'collision.csv.gz','rt') as f:c=list(csv.DictReader(f))
        axes[2,0].plot(series(c,'time_s')[::20],1000*series(c,'continuous_lower_m')[::20],label=mode,color=color)
        axes[2,1].plot(series(rows,'state_x'),series(rows,'state_y'),label=mode,color=color)
    axes[2,0].axhline(25,color='red',ls='--',lw=1,label='required 25 mm')
    for o in scene['obstacles']:
        xy=o['center'][:2];r=o['radius'];low=o['center'][2]-r<.35
        axes[2,1].add_patch(plt.Circle(xy,r,fill=low,alpha=.25,color='red',linestyle='-' if low else '--'))
        axes[2,1].text(*xy,o['name'],fontsize=7)
    labels=['Camera position error [mm]','Camera orientation error [deg]','Base v [m/s]',
            'Base omega [rad/s]','Continuous clearance lower bound [mm]','Base paths in world XY [m]']
    for i,(ax,label) in enumerate(zip(axes.flat,labels)):
        ax.set_title(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_xlabel('Time [s]' if i<5 else 'X [m]')
        if name=='combo' and i<5:
            for j in [22,36]:ax.axvline(j,color='k',ls=':',alpha=.4)
    axes[2,1].set_aspect('equal',adjustable='datalim');axes[2,1].set_ylabel('Y [m]')
    fig.suptitle(name+' | Isaac measurements (failures show only observed prefix)')
    fig.tight_layout();fig.savefig(out/f'{name}_comparison.png',dpi=150);plt.close(fig)
    if 'kc' in runs:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from jvs_mock.kc_svsdf import CollisionGeometry
        from jvs_mock.jvs_kinematics import forward_kinematics
        from scipy.spatial.transform import Rotation
        geometry=CollisionGeometry();rows=runs['kc'][1]
        chosen=[0,int(np.argmin(series(rows,'clearance_m'))),len(rows)-1]
        fig=plt.figure(figsize=(14,5))
        for panel,index in enumerate(chosen,1):
            ax=fig.add_subplot(1,3,panel,projection='3d');r=rows[index]
            x=np.array([float(r['state_'+k]) for k in ['x','y','yaw']+[f'q{i}' for i in range(1,7)]])
            arm_p,arm_r=forward_kinematics(x[3:])
            camera_r=Rotation.from_quat([float(r['actual_camera_q'+k]) for k in 'xyzw']).as_matrix()
            br=camera_r@arm_r.T
            z=float(r['actual_camera_z'])-(br@arm_p)[2]
            frames,_,_=geometry.frames(x,z,br)
            for body,_,kind,size,p,rot in geometry.shapes:
                bp,br=frames[body];center=bp+br@p;rotation=br@rot
                if kind in ('cylinder','frustum'):
                    rad0,rad1,height=(size[0],size[0],size[1]) if kind=='cylinder' else size
                    theta=np.linspace(0,2*np.pi,24)
                    for z,rad in [(-height/2,rad0),(height/2,rad1)]:
                        local=np.c_[rad*np.cos(theta),rad*np.sin(theta),np.full(len(theta),z)]
                        points=local@rotation.T+center
                        ax.plot(*points.T,color='#1677b8',alpha=.65,lw=.7)
                elif kind=='box':
                    import itertools
                    corners=np.array(list(itertools.product([-1,1],repeat=3)))*size/2
                    for i,a in enumerate(corners):
                        for b in corners[i+1:]:
                            if np.sum(abs(a-b)>1e-9)==1:
                                ax.plot(*((np.array([a,b])@rotation.T)+center).T,color='#1677b8',lw=.7)
            for o in scene['obstacles']:
                a,b=np.meshgrid(np.linspace(0,2*np.pi,18),np.linspace(0,np.pi,10));rad=o['radius'];c=o['center']
                ax.plot_surface(c[0]+rad*np.cos(a)*np.sin(b),c[1]+rad*np.sin(a)*np.sin(b),c[2]+rad*np.cos(b),color='tomato',alpha=.5)
            ax.set_title(f"KC t={float(r['time_s']):.2f}s, gap={float(r['clearance_m'])*1000:.1f}mm")
            ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]');ax.set_zlim(0,1.15)
            ax.set_xlim(x[0]-.45,x[0]+.45);ax.set_ylim(x[1]-.45,x[1]+.45);ax.set_box_aspect([.9,.9,1.15])
            ax.view_init(22,-60)
        fig.suptitle(name+' | collision geometry reconstructed from Isaac joint/base logs')
        fig.tight_layout();fig.savefig(out/f'{name}_configurations.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ros',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];all_results={};grouped={}
    for name in ['moveout','combo']:
        runs={}
        for mode in ['none','per_link','kc']:
            folder=next((f/f'{name}_{mode}' for f in args.folders if (f/f'{name}_{mode}'/'summary.json').exists()),None)
            if folder is None:continue
            stats,rows=statistics(folder);runs[mode]=(stats,rows,folder)
            all_results[f'{name}_{mode}']=stats
        if runs:
            common_end=min(max(float(r['time_s']) for r in rows if r['phase']=='tracking')
                           for _,rows,_ in runs.values() if rows)
            for stats,rows,_ in runs.values():
                common=[r for r in rows if r['phase']=='tracking' and float(r['time_s'])<=common_end]
                stats['common_tracking_until_s']=common_end
                stats['common_position_rmse_mm']=1000*float(np.sqrt(np.mean(series(common,'position_error')**2)))
                stats['common_orientation_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean(series(common,'orientation_error')**2))))
            scene=json.loads((root/'scenes'/f'{name}.json').read_text())
            figures(name,runs,scene,args.output)
            if 'kc' in runs and 'none' in runs:
                kr,nr=runs['kc'][1],runs['none'][1];n=min(len(kr),len(nr))
                kd=np.column_stack([series(kr,'state_'+k)[:n] for k in ['x','y']+[f'q{i}' for i in range(1,7)]])
                nd=np.column_stack([series(nr,'state_'+k)[:n] for k in ['x','y']+[f'q{i}' for i in range(1,7)]])
                stats=runs['kc'][0]
                stats['base_deviation_from_none_m']=float(np.max(np.linalg.norm(kd[:,:2]-nd[:,:2],axis=1)))
                stats['arm_deviation_from_none_rad']=float(np.max(np.linalg.norm(kd[:,2:]-nd[:,2:],axis=1)))
                stats['gates']['base_arm_coordination']=bool(stats['base_deviation_from_none_m']>.02 and stats['arm_deviation_from_none_rad']>.05 and stats['base_and_arm_near_obstacles_fraction']>0)
                clear_path=root/'results/stage2_final'/f'{name}_stage2'/'telemetry.csv'
                if clear_path.exists():
                    clear=read_rows(clear_path);nn=min(len(kr),len(clear))
                    kc_state=np.column_stack([series(kr,'state_'+k)[:nn] for k in ['x','y']+[f'q{i}' for i in range(1,7)]])
                    clear_state=np.column_stack([series(clear,'state_'+k)[:nn] for k in ['x','y']+[f'q{i}' for i in range(1,7)]])
                    stats['clear_room_reference']=str(clear_path)
                    stats['base_deviation_from_clear_stage2_m']=float(np.max(np.linalg.norm(kc_state[:,:2]-clear_state[:,:2],axis=1)))
                    stats['arm_deviation_from_clear_stage2_rad']=float(np.max(np.linalg.norm(kc_state[:,2:]-clear_state[:,2:],axis=1)))
                    stats['gates']['base_arm_coordination']=bool(stats['base_deviation_from_clear_stage2_m']>.02 and stats['arm_deviation_from_clear_stage2_rad']>.05 and stats['base_and_arm_near_obstacles_fraction']>0)
            grouped[name]=runs
    kc=[v for k,v in all_results.items() if k.endswith('_kc')]
    complete=len(all_results)==6
    decision=bool(complete and len(kc)==2 and all(all(r['gates'].values()) for r in kc))
    report=dict(results=all_results,all_six_reports=complete,stage3_pass=decision,
                thresholds=dict(clearance_mm=25,end_position_rmse_mm=20,end_orientation_rmse_rad=.035,
                                command_violation=1e-6,normalized_step=.05),
                scope='two deterministic Isaac scenes; no mesh-room safety certification')
    if args.ros and (args.ros/'summary.json').exists():
        # Preserve the trajectory result captured before intentional launch teardown.
        ros_summary=args.ros/'tracking_summary.json'
        if not ros_summary.exists():ros_summary=args.ros/'summary.json'
        report['ros']=json.loads(ros_summary.read_text())
        report['ros_summary_source']=str(ros_summary.resolve())
        ros_rows=read_rows(args.ros/'telemetry.csv')
        tracking=[r for r in ros_rows if r['phase']=='tracking']
        tail=[r for r in tracking if float(r['time_s'])>=float(tracking[-1]['time_s'])-1] if tracking else []
        rp=1000*float(np.sqrt(np.mean(series(tail,'position_error')**2))) if tail else None
        rr=float(np.rad2deg(np.sqrt(np.mean(series(tail,'orientation_error')**2)))) if tail else None
        ros_pass=bool(report['ros']['completed'] and tail and rp<=20 and rr<=np.rad2deg(.035))
        report['ros_stage3_functional']=dict(completed=report['ros']['completed'],end_position_rmse_mm=rp,
            end_orientation_rmse_deg=rr,recovery_pass=ros_pass,
            minimum_control_rate_clearance_mm=1000*min(series(ros_rows,'clearance_m')) if ros_rows else None,
            preview_age_p95_s=float(np.percentile(series(ros_rows,'preview_age_s'),95)) if ros_rows else None,
            preview_age_max_s=float(max(series(ros_rows,'preview_age_s'))) if ros_rows else None,
            fast_qp_p95_ms=1000*float(np.percentile(series(ros_rows,'solve_s'),95)) if ros_rows else None,
            controller_p95_ms=1000*float(np.percentile(series(ros_rows,'controller_s'),95)) if ros_rows else None,
            scope='ROS-rate FK feedback; independent 1 kHz physics certificate remains the lockstep suite')
        decision=decision and ros_pass
        report['stage3_pass']=decision
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Stage 3 Isaac comparison','',f'**Stage 3 strict acceptance: {"PASS" if decision else "NOT PASSED"}.**','',
        'Minimum safety clearance: 25 mm. All rates include every recorded 1 ms physics state; an aborted run covers only its observed prefix. Negative clearance means conservative-envelope intersection; this is not a separate PhysX contact counter. One deterministic trial per method/scenario; no statistical success-rate claim.','',
        '| Scene / mode | Complete | Collision states | Swept lower [mm] | Pos RMS [mm] | Rot RMS [deg] | MPC mean / P95 [ms] | End pos RMS [mm] |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for key,r in all_results.items():
        lines.append(f"| {key} | {r['completed']} | {r['collision_state_rate']:.2%} | {r['continuous_lower_mm']:.2f} | {r['position_rmse_mm']:.2f} | {r['orientation_rmse_deg']:.2f} | {r['mpc_mean_ms']:.1f} / {r['mpc_p95_ms']:.1f} | {r['end_position_rmse_mm']:.2f} |")
    lines+=['','## Acceptance details','']
    for name,runs in grouped.items():
        lines += [f"Common tracking window for {name}: 0–{next(iter(runs.values()))[0]['common_tracking_until_s']:.2f} s. Position/orientation RMS: "+'; '.join(
            f"{mode} {r['common_position_rmse_mm']:.2f} mm / {r['common_orientation_rmse_deg']:.2f} deg" for mode,(r,_,_) in runs.items())+'.', '']
    for name,runs in grouped.items():
        if 'kc' not in runs:continue
        r=runs['kc'][0]
        lines += [f'### {name}', '', f"Gates: `{json.dumps(r['gates'])}`.", '',
            f"Peak camera error {r['position_peak_mm']:.2f} mm; end orientation RMS {r['end_orientation_rmse_deg']:.3f} deg. Maximum normalized control step {r['max_normalized_control_step']:.5f}; command residual {r['max_command_violation']:.3g}; planner failures {r['planner_failures']}.", '',
            f"Relative to no avoidance: base deviation {r.get('base_deviation_from_none_m',0):.3f} m, arm configuration deviation {r.get('arm_deviation_from_none_rad',0):.3f} rad. Simultaneous nearby rigid bodies: {r['max_simultaneous_bodies_near_obstacles']}; base and arm both within 100 mm of obstacles during {r['base_and_arm_near_obstacles_fraction']:.1%} of recorded physics states. These differences show coordinated response, not an optimality guarantee.", '',
            f"Unfiltered physical maxima: `{json.dumps(r['physical_maxima'])}`.", '',f'![{name} comparison]({name}_comparison.png)','',
            f'![{name} configurations]({name}_configurations.png)','']
        if 'clear_room_reference' in r:
            lines += [f"Relative to the same Stage 2 trajectory in an empty scene: base deviation {r['base_deviation_from_clear_stage2_m']:.3f} m, arm configuration deviation {r['arm_deviation_from_clear_stage2_rad']:.3f} rad. This avoids attributing the colliding baseline's physical displacement to active avoidance.", '']
        if 'junctions' in r:
            lines += [f"Within ±0.2 s of each shot junction (successive 20 ms samples): `{json.dumps(r['junctions'])}`.", '']
    failures=[f"- {k}: {r['failure']}" for k,r in all_results.items() if r['failure']]
    if failures:lines+=['## Recorded failures','',*failures,'']
    if 'ros_stage3_functional' in report:
        lines+=['## ROS–Isaac integration', '',f"`{json.dumps(report['ros_stage3_functional'])}`", '',
                f"Full-run position RMS {report['ros']['position_error_rms']*1000:.2f} mm; orientation RMS {np.rad2deg(report['ros']['orientation_error_rms']):.2f} deg. Source: `{report['ros_summary_source']}`.", '',
                f"Reported fault: `{report['ros'].get('failure','')}`. Global RMS retains transient avoidance errors; end recovery is evaluated separately. This run does not replace the independent physical-step collision audit.", '']
    if (args.output/'combo_kc_replay.mp4').exists():
        lines+=['[Full combo KC telemetry replay (MP4, 3x)](combo_kc_replay.mp4)', '',
                'The video reconstructs measured Isaac joint/base/camera motion; it is a scientific replay, not a recording of the Isaac renderer.', '']
    if (args.output/'physical_acceleration_diagnostics.json').exists():
        lines+=['[Per-channel physical acceleration peaks, limits and timestamps](physical_acceleration_diagnostics.json). These are unsmoothed adjacent 1 ms velocity differences; linear and angular units are kept separate.', '']
    lines+=['## Interpretation','',
        'Both avoidance methods retain Stage 2 shot inputs/preferences and use the same route seed, cost weights, hard actuator bounds, clearance CBF and command sweep check. Only the KC upstream candidate filter differs. Neither a prefix with no collisions nor elapsed timeline completion alone proves successful tracking recovery.', '',
        'PhysX pose auditing bounds translation/SLERP interpolation between every adjacent measured rigid-body pose. It includes all URDF collision geometry, with conservative chassis/caster enclosures. It excludes support ground contact and self-collision. Predictive/ROS timing is measured, not a hard real-time guarantee.', '',
        'The user-provided simview room was inspected in Isaac separately. Its meshes/animated objects are not in the sphere-only controller map, so no room collision-free completion is claimed. See ../stage3_simview/inspection.json.', '',
        'Development results under stage3_dev* are excluded from this report. The input folders printed below are the authoritative evaluated runs:', '',*[f'- {str(f.resolve())}' for f in args.folders]]
    if (args.output/'PROVENANCE.md').exists():
        lines+=['', '[Run provenance and excluded development trials](PROVENANCE.md).', '']
    if (args.output/'source_manifest.json').exists():
        lines+=['[Source hashes and final-run hash verification](source_manifest.json).', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage3_pass=decision,reports=len(all_results),kc=[r['gates'] for r in kc])))

if __name__=='__main__':main()
