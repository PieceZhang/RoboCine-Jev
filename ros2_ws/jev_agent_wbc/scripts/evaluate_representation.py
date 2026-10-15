#!/usr/bin/env python3
"""Matched-state geometry ablation on six existing demos (not a safety rollout).

Truth obstacle poses are used ONLY here for offline evaluation. Nominal snapshots
or recorded telemetry drive all variants identically. No Agent receives truth.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from jev_agent_wbc.representation import Box,PARTS
from jev_agent_wbc.representation_jvs import JVSGeometryBackend
from jev_agent_wbc.scenarios import RuntimeScene,NAMES
from jev_agent_wbc.reference import OriginalReference
from jev_agent_wbc.nominal import nominal_rollout
from jev_agent_wbc.jev import compact_evidence

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--telemetry-root',type=Path)
    parser.add_argument('--snapshots',type=int,default=17)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261015)
    rows=[]
    for case in NAMES:
        runtime=RuntimeScene(ROOT/'scenarios'/case/'runtime_scene.json')
        fixture=runtime.document;z=fixture['base_z']
        reference=OriginalReference(ROOT/'scenarios'/case/'task.json',fixture['subject']['position'])
        if args.telemetry_root:
            telemetry=json.loads((args.telemetry_root/f'{case}_jev/telemetry.json').read_text())
            times=np.array([r['t'] for r in telemetry]);states=np.array([r['state'] for r in telemetry]);failure=''
            source='recorded_CPU_execution'
        else:
            times,states,failure=nominal_rollout(reference,fixture['initial_state'],z)
            source='offline_nominal_FK_snapshots_not_obstacle_avoidance_execution'
        backends={budget:JVSGeometryBackend(budget) for budget in (1,3,6)}
        for index in np.linspace(0,len(states)-1,min(args.snapshots,len(states)),dtype=int):
            state,t=states[index],float(times[index])
            geometry=backends[1].geometry
            mount,_,raw=backends[1].primitive_boxes(state,z)
            bounds=Box.enclosing(np.vstack([b.vertices for boxes in raw.values() for b in boxes]))
            local=rng.uniform(-1,1,(2500,3))*(np.array(bounds.half_extents_m)+.2)+bounds.center_m
            world=mount.apply(local)
            exact_by_body=geometry.distances(state,world,z)
            exact=np.c_[exact_by_body[:,0],exact_by_body[:,1:].min(axis=1)]
            obstacles=runtime.obstacles(t)
            obstacle_world=np.array([o['center'] for o in obstacles]);radii=np.array([o['radius'] for o in obstacles])
            obstacle_local=mount.inverse().apply(obstacle_world)
            exact_obstacle=geometry.distances(state,obstacle_world,z)
            exact_obstacle=np.c_[exact_obstacle[:,0],exact_obstacle[:,1:].min(axis=1)]-radii[:,None]
            for budget,backend in backends.items():
                started=time.perf_counter();obs=backend.observe(state,t,base_z=z,source='synthetic_fk')
                elapsed=(time.perf_counter()-started)*1000
                proxy=np.array([np.min([b.distance(local) for b in obs.occupancy[part]],axis=0) for part in PARTS]).T
                proxy_obstacle=np.array([np.min([b.distance(obstacle_local) for b in obs.occupancy[part]],axis=0) for part in PARTS]).T-radii[:,None]
                free=exact>=.025
                text=json.dumps(compact_evidence(dict(robot_geometry=obs.summary())),separators=(',',':'))
                rows.append(dict(case=case,time_s=t,box_budget=budget,snapshot_source=source,
                    snapshot_generation_failure=failure,build_ms=elapsed,text_bytes=len(text.encode()),
                    false_occupied_count=int(((proxy<.025)&free).sum()),free_probe_count=int(free.sum()),
                    missed_occupied_count=int(((exact<.025)&(proxy>=.025)).sum()),
                    clearance_overestimate_max_m=float(np.max(proxy-exact)),
                    obstacle_clearance_underestimate_max_m=float(np.max(exact_obstacle-proxy_obstacle)),
                    exact_obstacle_min_m=float(exact_obstacle.min()),proxy_obstacle_min_m=float(proxy_obstacle.min())))
        print(json.dumps(dict(case=case,snapshots=min(args.snapshots,len(states)),source=source)),flush=True)
    aggregate=[]
    for budget in (1,3,6):
        subset=[r for r in rows if r['box_budget']==budget]
        aggregate.append(dict(box_budget=budget,snapshots=len(subset),
            false_occupied_fraction=sum(r['false_occupied_count'] for r in subset)/sum(r['free_probe_count'] for r in subset),
            missed_occupied_count=sum(r['missed_occupied_count'] for r in subset),
            text_bytes_mean=float(np.mean([r['text_bytes'] for r in subset])),
            build_ms_mean=float(np.mean([r['build_ms'] for r in subset])),
            build_ms_p95=float(np.quantile([r['build_ms'] for r in subset],.95)),
            obstacle_clearance_underestimate_mean_m=float(np.mean([r['obstacle_clearance_underestimate_max_m'] for r in subset])),
            clearance_overestimate_max_m=max(r['clearance_overestimate_max_m'] for r in subset)))
    (args.output/'geometry_ablation.json').write_text(json.dumps(dict(seed=20261015,rows=rows,aggregate=aggregate),indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))
    return int(any(r['missed_occupied_count'] for r in aggregate))


if __name__=='__main__':raise SystemExit(main())
