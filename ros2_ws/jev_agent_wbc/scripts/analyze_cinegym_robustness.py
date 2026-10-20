#!/usr/bin/env python3
"""Offline scene/paired-run audit; truth never enters controller decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CineGym.episode import Episode, write_json
from CineGym.evaluate import summarize_runs
from jev_agent_wbc.kc_svsdf import CollisionGeometry, advance
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.self_collision import SelfCollision


def stats(values):
    x = np.asarray(values, float)
    if not len(x):
        return None
    return dict(n=len(x), minimum=float(x.min()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), maximum=float(x.max()))


def scenes(root,suites=None,example_suite='train'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = {}
    for suite in (suites or ('train', 'test', 'calibrated_train', 'calibrated_test','longer_train','final_test')):
        if not (root/suite/'catalog.json').exists():continue
        episodes = [Episode.load(p.parent) for p in sorted((root/suite).glob('*/metadata.json'))]
        rows = [o for e in episodes for o in e.metadata['audit']['obstacles']]
        groups = {}
        for difficulty in ('easy', 'medium'):
            chosen = [e for e in episodes if e.metadata['difficulty'] == difficulty]
            obstacles = [o for e in chosen for o in e.metadata['audit']['obstacles']]
            conflicts=[o for o in obstacles if o.get('interaction_role','conflict')=='conflict']
            passers=[o for o in obstacles if o.get('interaction_role')=='passing']
            cap = .035 if difficulty == 'easy' else .065
            groups[difficulty] = dict(episodes=len(chosen), obstacles=len(obstacles),
                episodes_above_global_penetration_profile=sum(
                    e.metadata['audit']['min_surface_clearance_m'] < -cap-1e-7 for e in chosen),
                conflict_actors=len(conflicts),passing_actors=len(passers),
                penetration_m=stats([-o['min_surface_clearance_m'] for o in conflicts]),
                passing_min_clearance_m=stats([o['min_surface_clearance_m'] for o in passers]),
                first_conflict_s=stats([o['first_conflict_s'] for o in conflicts if o.get('first_conflict_s') is not None]),
                conflict_duration_s=stats([o['conflict_duration_s'] for o in conflicts if 'conflict_duration_s' in o]),
                continuous_speed_bound_m_s=stats([o['continuous_motion_bounds']['speed_m_s'] for o in obstacles if 'continuous_motion_bounds' in o]),
                continuous_acceleration_bound_m_s2=stats([o['continuous_motion_bounds']['acceleration_m_s2'] for o in obstacles if 'continuous_motion_bounds' in o]),
                continuous_jerk_bound_m_s3=stats([o['continuous_motion_bounds']['jerk_m_s3'] for o in obstacles if 'continuous_motion_bounds' in o]),
                max_speed_m_s=stats([o['max_speed_m_s'] for o in obstacles]),
                max_acceleration_m_s2=stats([o['max_acceleration_m_s2'] for o in obstacles]))
        result[suite] = dict(episodes=len(episodes), obstacles=len(rows), groups=groups,
            shapes=dict(Counter(o.get('shape','sphere') for e in episodes for o in e.scene['obstacles'])),
            catalog_complete=json.loads((root/suite/'catalog.json').read_text())['complete'],
            nominal_shot_accepted=sum(bool((e.metadata.get('nominal_shot_acceptance') or {}).get('accepted')) for e in episodes),
            avoidance_solvability=dict(Counter(e.metadata.get('avoidance_solvability',{}).get('status','unverified') for e in episodes)),
            motions=dict(Counter(o['motion'] for o in rows)),
            heights=dict(Counter(o['height_level'] for o in rows)),
            action_counts=dict(Counter(len(e.task['shot_phase']) for e in episodes)),
            simultaneous_interferences=stats([e.metadata['audit']['max_simultaneous_interferences'] for e in episodes]),
            initial_clearance_m=stats([o['initial_clearance_m'] for o in rows]))
    incomplete = root/'warning_train'/'catalog.json'
    if incomplete.exists():
        c = json.loads(incomplete.read_text())
        result['long_warning_generation'] = dict(complete=c['complete'], generated=len(c['cases']),
            requested=c['requested_count'], excluded_from_comparisons=True)
    write_json(root/'scene_assessment.json', result)
    fig, axes = plt.subplots(2, 6, figsize=(19, 7), constrained_layout=True)
    cases = sorted((root/example_suite).glob('*/metadata.json'))
    colors = {'low':'#d97706', 'middle':'#2563eb', 'high':'#9333ea'}
    for col, index in enumerate((0, 3, 4, 7, 14, 19)):
        e = Episode.load(cases[index].parent)
        t = np.linspace(0, e.reference.duration, 301)
        camera = np.array([e.reference.sample(float(at))[0] for at in t])
        frames = CollisionGeometry().frames(e.scene['initial_state'], e.scene['base_z'])[0]
        body = np.array([p for p, _ in frames])
        for row, dims in enumerate(((0, 1), (0, 2))):
            ax = axes[row, col]
            ax.plot(camera[:,dims[0]], camera[:,dims[1]], 'k--', lw=1.7, label='Camera reference')
            ax.plot(body[:,dims[0]], body[:,dims[1]], 'o-', color='#64748b', ms=3, label='Initial body origins')
            for obj, audit in zip(e.scene['obstacles'], e.metadata['audit']['obstacles']):
                p = e.runtime.curves[obj['id']](t)
                color = colors[audit['height_level']]
                if obj.get('shape')=='box':
                    import itertools
                    from scipy.spatial import ConvexHull
                    from jev_agent_wbc.runtime_obstacles import box_rotation
                    vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))*np.array(obj['size'])/2
                    vertices=vertices@box_rotation(obj).T+p[0]
                    polygon=vertices[:,list(dims)];hull=ConvexHull(polygon)
                    ax.add_patch(plt.Polygon(polygon[hull.vertices],color=color,alpha=.25,label='Ground box'))
                    continue
                ax.plot(p[:,dims[0]], p[:,dims[1]], color=color, alpha=.8,
                    linestyle='--' if audit.get('interaction_role')=='passing' else '-',label=audit['motion'])
                ax.add_patch(plt.Circle(p[0,list(dims)], obj['radius'], color=color, alpha=.2))
                ax.scatter(*p[-1,list(dims)], marker='x', color=color, s=20)
            ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2)
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)' if row==0 else 'z (m)')
            if row==0:
                ax.set_title(f"{e.metadata['seed']} / {e.metadata['difficulty']}\n{e.metadata['family']}")
    axes[0,0].legend(fontsize=7, loc='best')
    fig.suptitle('Random scenes: full obstacle paths for offline inspection only; circles=start, crosses=end')
    fig.savefig(root/'random_scene_examples.png', dpi=160)
    plt.close(fig)
    return result


def curve_speed_bound(curve):
    # The velocity quadratic lies in the convex hull of its Bernstein controls.
    h = np.diff(curve.x)[:,None]
    a, b, c = 3*curve.c[0], 2*curve.c[1], curve.c[2]
    control = np.stack([c, c+b*h/2, c+b*h+a*h*h])
    return float(np.linalg.norm(control, axis=-1).max())


def audit_run(folder, step=.005):
    rows = json.loads((folder/'telemetry.json').read_text()) if (folder/'telemetry.json').exists() else []
    runtime = RuntimeScene(folder/'runtime_scene.json')
    geometry, own = CollisionGeometry(), SelfCollision()
    failure = folder/'failure_snapshot.json'
    endpoint = json.loads(failure.read_text()) if failure.exists() else None
    endpoint_source='recorded' if endpoint else None
    summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    if endpoint is None and str(summary.get('failure','')).startswith('RuntimeError: TPR'):
        # Historical runner snapshots predate failure_snapshot.json. A TPR
        # exception is raised inside engine.step before the next telemetry row;
        # the previous recorded constant command was already integrated.
        endpoint=dict(time_s=rows[-1]['t']+.02 if rows else 0.,
            state=advance(np.asarray(rows[-1]['state']),np.asarray(rows[-1]['command']),.02).tolist()
                  if rows else runtime.document['initial_state'])
        endpoint_source='reconstructed_last_constant_command_before_TPR_failure'
    if endpoint is None and rows and summary.get('completed'):
        from jev_agent_wbc.camera_intent import POLICY
        # CPU runner records BEFORE integration. Its final rounded sample can
        # still satisfy the strict t < duration + hold branch (including a
        # floating-point boundary), which executes one more 20 ms command.
        # Include that exact command rather than certifying only the log prefix.
        end=summary.get('original_duration_s')
        if end is not None and rows[-1]['t'] < end+POLICY.terminal_hold_s:
            endpoint=dict(time_s=rows[-1]['t']+.02,
                state=advance(np.asarray(rows[-1]['state']),np.asarray(rows[-1]['command']),.02).tolist())
            endpoint_source='reconstructed_final_constant_command_after_last_telemetry'
    times, states, speeds, self_speeds = [], [], [], []
    max_gap = 0.
    for i, row in enumerate(rows):
        end = rows[i+1]['t'] if i+1<len(rows) else row['t']
        if i+1 == len(rows) and endpoint is not None:
            end = endpoint['time_s']
        duration = max(0.,end-row['t'])
        count = max(1,int(np.ceil(duration/step)))
        offsets = np.linspace(0.,duration,count+1)[:-1] if duration else np.array([0.])
        max_gap = max(max_gap,duration/count)
        u = np.asarray(row['command'])
        # Runner integrates each command as constant velocity, not a ramp.
        for offset in offsets:
            times.append(row['t']+offset)
            states.append(advance(np.asarray(row['state']),u,float(offset)))
            speeds.append(abs(u[0])+abs(u[1:])@geometry.reach)
            self_speeds.append(2*abs(u[2:])@geometry.reach[1:])
    if endpoint is not None:
        f = endpoint
        if not times or f['time_s']>times[-1]+1e-10:
            times.append(f['time_s']); states.append(f['state'])
            speeds.append(speeds[-1] if speeds else 0.); self_speeds.append(self_speeds[-1] if self_speeds else 0.)
    if not states:
        return dict(samples=0, status='unverified')
    times, states = np.asarray(times), np.asarray(states)
    centers=[];radii=[]
    for obj in runtime.document['obstacles']:
        curve=runtime.curves[obj['id']];offsets,sizes=runtime.collision_covers[obj['id']]
        centers.append(curve(np.clip(times,curve.x[0],curve.x[-1]))[:,None]+offsets)
        radii.extend(sizes)
    centers=np.concatenate(centers,axis=1);radii=np.array(radii)
    clear, self_clear = [], []
    for start in range(0,len(states),300):
        batch = states[start:start+300]
        distances = geometry.distances_sequence(batch,centers[start:start+300],runtime.document['base_z'])
        clear.extend((distances-radii[None,:,None]).min(axis=(1,2)).tolist())
        self_clear.extend(own.distances_sequence(batch).min(axis=1).tolist())
    clear, self_clear = np.asarray(clear), np.asarray(self_clear)
    obstacle_bound = max(curve_speed_bound(c) for c in runtime.curves.values())
    lower = float(clear.min()-(max(speeds)+obstacle_bound)*max_gap/2)
    own_lower = float(self_clear.min()-max(self_speeds)*max_gap/2)
    return dict(samples=len(times), end_s=float(times[-1]), max_sample_gap_s=max_gap,
        failure_endpoint_source=endpoint_source,
        min_truth_clearance_m=float(clear.min()), min_self_lower_m=float(self_clear.min()),
        margin_violation_samples=int(sum(clear<.025)), penetration_samples=int(sum(clear<0.)),
        self_margin_violation_samples=int(sum(self_clear<own.margin)),
        continuous_external_lower_m=lower, continuous_self_lower_m=own_lower,
        continuous_safety_certified=lower>=.025 and own_lower>=own.margin,
        scope='Recorded execution only, including final failed state; exact constant-command integration; '
              '5 ms samples plus global velocity Lipschitz bound. Does not certify remaining unexecuted shot.')


def benchmark(root, name, dense=False):
    folder = root/name
    report = summarize_runs(folder)
    latencies, events, rejection, loop, pacing = [], Counter(), Counter(), [], []
    response_counts={}
    replan_max, preview_accept, preview_reject, installed = [], [], [], 0
    motion_errors, motion_samples, motion_matched = [], 0, 0
    audit = {}
    for runtime_file in sorted(folder.glob('*/*/*/runtime_scene.json')):
        path = runtime_file.parent
        summary=path/'summary.json'
        s = json.loads(summary.read_text()) if summary.exists() else {}
        installed += s.get('installed_plans',0)
        if s.get('elapsed_shot_s',0)>0:
            pacing.append(s['wall_s']/s['elapsed_shot_s'])
        before=len(latencies)
        for p in path.glob('*jev_api.jsonl'):
            for line in p.read_text().splitlines():
                d=json.loads(line)
                if d.get('response') is not None and not d.get('error'):
                    latencies.append(d['wall_s'])
        response_counts[path.parent.name]=len(latencies)-before
        if (path/'events.jsonl').exists():
            for line in (path/'events.jsonl').read_text().splitlines():
                d=json.loads(line); events[d['event']]+=1
                for e in d.get('evaluations') or []:
                    if e.get('rejected'): rejection[e['rejected']]+=1
        telemetry = json.loads((path/'telemetry.json').read_text()) if (path/'telemetry.json').exists() else []
        loop.extend(r['info']['control_thread_wall_s']*1000 for r in telemetry if 'control_thread_wall_s' in r['info'])
        replan_max.append(max((r['info'].get('tpr_motion_replans',0) for r in telemetry),default=0))
        preview_accept.append(max((r['info'].get('tpr_prediction_accepted',0) for r in telemetry),default=0))
        preview_reject.append(max((r['info'].get('tpr_prediction_rejected',0) for r in telemetry),default=0))
        # Compare each distinct map acquisition against truth at THAT time.
        # Nearest associations are for this offline audit only.
        from scipy.optimize import linear_sum_assignment
        runtime=RuntimeScene(path/'runtime_scene.json');seen=set()
        for row in telemetry:
            stamp=row['observation_stamp']
            if stamp in seen:continue
            seen.add(stamp)
            truth=runtime.obstacles(stamp)
            moving=[o for o in truth if np.linalg.norm(o['velocity'])>.025]
            motion_samples+=len(moving)
            tracks=row.get('runtime_tracks') or []
            if not tracks or not moving:continue
            centers=np.array([o['center'] for o in moving]);measured=np.array([o['center'] for o in tracks])
            distances=np.linalg.norm(centers[:,None]-measured[None],axis=-1)
            for a,b in zip(*linear_sum_assignment(distances)):
                if distances[a,b]>.15:continue
                motion_matched+=1
                motion_errors.append(np.linalg.norm(np.asarray(tracks[b]['velocity'])-moving[a]['velocity']))
        if dense:
            audit[path.parent.name] = audit_run(path)
    report.update(api_answer_latency_s=stats(latencies), events=dict(events),
                  cases_with_official_responses=sum(n>0 for n in response_counts.values()),
                  cases_without_official_responses=[name for name,n in response_counts.items() if n==0],
                  candidate_rejections=dict(rejection), installed_plans_total=installed,
                  control_wall_ms=stats(loop), wall_sim_ratio=stats(pacing),
                  motion_triggered_replans_total=int(sum(replan_max)),
                  accepted_local_previews_total=int(sum(preview_accept)),
                  rejected_local_previews_total=int(sum(preview_reject)),
                  observed_motion_audit=dict(truth_moving_object_acquisitions=motion_samples,
                      associated_within_15cm=motion_matched,velocity_error_m_s=stats(motion_errors),
                      scope='Offline nearest one-to-one association at map acquisition time; includes unconfirmed zero-velocity tracks; no future truth used by agent'),
                  dense_audits=audit,
                  source_sha256={str(p.relative_to(folder/'frozen_code')):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((folder/'frozen_code').rglob('*.py'))})
    write_json(root/(name+'_analysis.json'),report)
    return report


def paired(a, b):
    left = {r['seed']:r for r in a['runs'] if r['status']!='unverified'}
    right = {r['seed']:r for r in b['runs'] if r['status']!='unverified'}
    seeds = sorted(left.keys() & right.keys())
    for s in seeds:
        assert left[s]['episode_sha256']==right[s]['episode_sha256'], 'paired inputs differ'
    def measure(rows):
        return dict(n=len(rows), full_success=sum(r['status']=='pass' for r in rows),
            safety_pass=sum(r['safety_pass'] is True for r in rows),
            safe_completed=sum(r['completed'] and r['safety_pass'] is True for r in rows),
            completed=sum(r['completed'] for r in rows))
    return dict(common_verified_seeds=seeds, baseline=measure([left[s] for s in seeds]),
                optimized=measure([right[s] for s in seeds]),
                excluded_for_network_or_missing_evidence=sorted((left.keys() | right.keys())-set(seeds)))


def merge_network_retries(reports):
    """Fixed rule: resolve explicitly unverified trials, never replace a failure.

    Includes transport failures and recorded wall-limit/external interruptions;
    each retry must use identical controller/input and be predeclared separately.
    """
    records={};history={}
    for report in reports:
        for record in report['runs']:
            row=dict(record);seed=row['seed']
            row['dense_audit']=report.get('dense_audits',{}).get(Path(row['output']).name)
            history.setdefault(seed,[]).append(dict(output=row['output'],status=row['status']))
            if seed in records:
                assert records[seed]['episode_sha256']==row['episode_sha256']
                assert records[seed]['status']=='unverified', 'cannot select a better algorithm outcome'
            records[seed]=row
    return dict(runs=[records[s] for s in sorted(records)],attempt_history=history)


def comparison(root, name, left_names, right_names):
    import csv
    def collect(names):
        return merge_network_retries([json.loads((root/(n+'_analysis.json')).read_text()) for n in names])
    left,right=collect(left_names),collect(right_names)
    left_by_seed={r['seed']:r for r in left['runs']}
    for row in right['runs']:
        if row['seed'] in left_by_seed:
            original=left_by_seed[row['seed']]
            assert all(row[k]==original[k] for k in ('episode_sha256','family','difficulty'))
    def metrics(rows):
        verified=[r for r in rows if r['status']!='unverified']
        audited=[r for r in verified if r.get('dense_audit') and r['dense_audit'].get('samples',0)]
        def safe(r):
            a=r['dense_audit'];return a['margin_violation_samples']==0 and a['self_margin_violation_samples']==0
        return dict(total=len(rows),verified=len(verified),unverified=len(rows)-len(verified),
            reported_full_success=sum(r['status']=='pass' for r in verified),
            full_success=sum(r['status']=='pass' and safe(r) for r in audited),
            completed=sum(r['completed'] for r in verified),audited=len(audited),
            safe_completed=sum(r['completed'] and safe(r) for r in audited),
            prefix_no_margin_violation=sum(safe(r) for r in audited),
            prefixes_with_penetration=sum(r['dense_audit']['penetration_samples']>0 for r in audited),
            continuous_safe_prefixes=sum(r['dense_audit']['continuous_safety_certified'] for r in audited),
            continuous_safe_completed=sum(r['completed'] and r['dense_audit']['continuous_safety_certified'] for r in audited),
            continuous_full_success=sum(r['status']=='pass' and r['dense_audit']['continuous_safety_certified'] for r in audited))
    common=sorted({r['seed'] for r in left['runs'] if r['status']!='unverified'} &
                  {r['seed'] for r in right['runs'] if r['status']!='unverified'})
    result=dict(baseline=metrics(left['runs']),optimized=metrics(right['runs']),
                common_verified_seeds=common,
                paired_baseline=metrics([r for r in left['runs'] if r['seed'] in common]),
                paired_optimized=metrics([r for r in right['runs'] if r['seed'] in common]),
                by_difficulty={},attempts=dict(baseline=left['attempt_history'],optimized=right['attempt_history']))
    for difficulty in ('easy','medium'):
        result['by_difficulty'][difficulty]=dict(
            baseline=metrics([r for r in left['runs'] if r['difficulty']==difficulty]),
            optimized=metrics([r for r in right['runs'] if r['difficulty']==difficulty]))
    with (root/(name+'_comparison.csv')).open('w',newline='') as stream:
        fields=['variant','seed','family','difficulty','status','completed','safety_pass','installed_plans',
                'min_truth_clearance_m','continuous_external_lower_m','continuous_self_lower_m','shot_failed_checks','output']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for variant,report in (('baseline',left),('optimized',right)):
            for row in report['runs']:
                item={k:row.get(k) for k in fields};item['variant']=variant
                item.update({k:(row.get('dense_audit') or {}).get(k) for k in
                             ('min_truth_clearance_m','continuous_external_lower_m','continuous_self_lower_m')})
                writer.writerow(item)
    write_json(root/(name+'_comparison.json'),result)
    return result


def plot_comparisons(root, stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[];counts=[]
    for title,left_names,right_names in stages:
        for variant,names in (('Baseline',left_names),('Optimized',right_names)):
            report=merge_network_retries([json.loads((root/(n+'_analysis.json')).read_text()) for n in names])
            count=np.zeros(5,dtype=int)
            for row in report['runs']:
                a=row.get('dense_audit') or {}
                if row['status']=='unverified' or not a.get('samples'):category=4
                elif a['margin_violation_samples'] or a['self_margin_violation_samples']:category=3
                elif row['status']=='pass':category=0
                elif row['completed']:category=1
                else:category=2
                count[category]+=1
            labels.append(title+'\n'+variant);counts.append(count)
    counts=np.array(counts);fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    colors=['#16a34a','#3b82f6','#94a3b8','#f97316','#a855f7']
    names=['Full shot success','Complete; shot quality failed','Stopped; executed prefix safe',
           'Safety-margin violation','Unverified']
    bottom=np.zeros(len(counts))
    for j,(color,label) in enumerate(zip(colors,names)):
        ax.bar(np.arange(len(counts)),counts[:,j],bottom=bottom,color=color,label=label,width=.65)
        for i,value in enumerate(counts[:,j]):
            if value:ax.text(i,bottom[i]+value/2,str(value),ha='center',va='center',fontsize=11)
        bottom+=counts[:,j]
    ax.set_xticks(np.arange(len(labels)),labels);ax.set_ylabel('Independent seeds')
    ax.set_ylim(0,max(bottom)*1.15);ax.spines[['top','right']].set_visible(False)
    ax.set_title('CPU closed loop with official Jev API: task success and safety are separate')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False,fontsize=9)
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--scenes',action='store_true')
    p.add_argument('--scene-suites',nargs='*');p.add_argument('--example-suite',default='train')
    p.add_argument('--groups',nargs='*',default=[]);p.add_argument('--dense',action='store_true')
    args=p.parse_args()
    if args.scenes:
        scenes(args.root,args.scene_suites,args.example_suite);print('scene audit complete',flush=True)
    for name in args.groups:
        report=benchmark(args.root,name,args.dense)
        print(name,json.dumps(report['groups'].get('cpu/all')),flush=True)


if __name__=='__main__':
    main()
