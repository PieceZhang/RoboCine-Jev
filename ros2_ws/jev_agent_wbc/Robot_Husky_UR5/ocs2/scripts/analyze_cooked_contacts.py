#!/usr/bin/env python3
"""Compare preserved native cooked meshes, source hulls and actual contacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROBOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROBOT/'results/ocs2_contact_diagnostic_01')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    run=args.run;model=json.loads((run/'sources/live_model.json').read_text())
    source={s['name']:s for s in model['collision_shapes']}
    bodies={b['name']:np.array(b['T_base_body']) for b in model['body_kinematics']}
    capture=json.loads((run/'cooked_collision_representation.json').read_text())
    converted=[];metrics=[]
    for item in capture['shapes']:
        if not item['convexes']:continue
        shape=source[item['path']];raw=np.array(item['source_mesh_points']);native=np.array(shape['vertices_body'])
        world=np.array(item['T_world_usd_prim']);base_body=bodies[shape['body']]
        # The runtime's authored base rotation is identity. Recover the exact
        # body<-mesh affine transform, including any authored mesh scale; its
        # translation is verified against BOTH bounds of the preserved hull.
        A=base_body[:3,:3].T@world[:3,:3];raw_rotated=raw@A.T
        translation=native.min(0)-raw_rotated.min(0)
        raw_body=raw_rotated+translation
        alignment=float(max(np.max(np.abs(raw_body.min(0)-native.min(0))),np.max(np.abs(raw_body.max(0)-native.max(0)))))
        if alignment>1e-6:raise ValueError('unverified cooked/source body frame '+item['path'])
        local=np.concatenate([np.array(c['vertices']) for c in item['convexes']])
        cooked_body=local@A.T+translation;cooked_world=local@world[:3,:3].T+world[:3,3]
        hull=ConvexHull(native);signed=cooked_body@hull.equations[:,:3].T+hull.equations[:,3]
        outside=float(max(0.,signed.max()))
        body_prim=np.eye(4);body_prim[:3,:3]=A;body_prim[:3,3]=translation
        out=dict(name=item['path'],body=shape['body'],T_body_prim=body_prim.tolist(),T_world_body_at_query=(world@np.linalg.inv(body_prim)).tolist(),raw_body_bounds=[native.min(0).tolist(),native.max(0).tolist()],cooked_body_bounds=[cooked_body.min(0).tolist(),cooked_body.max(0).tolist()],cooked_vertices_body=cooked_body.tolist(),cooked_vertices_world_at_query=cooked_world.tolist(),coordinate_scope='native cooking representation in mesh-local coordinates; body<-mesh transform including scale verified against source hull bounds',raw_body_frame_alignment_error_m=alignment,max_cooked_vertex_outside_raw_hull_m=outside)
        converted.append(out);metrics.append({k:v for k,v in out.items() if k not in ['cooked_vertices_body','cooked_vertices_world_at_query']})
    (args.output/'cooked_body_world_geometry.json').write_text(json.dumps(converted,indent=2)+'\n')
    contacts=[];counts=Counter();peaks={};minsep={};self_contacts=[]
    for line in (run/'contacts.jsonl').open():
        row=json.loads(line);pair=tuple(row['colliders']);counts[pair]+=1
        impulse=max([float(np.linalg.norm(p['impulse'])) for p in row['points']],default=0.)
        if impulse>peaks.get(pair,{}).get('maximum_point_impulse_Ns',-1.):peaks[pair]=dict(maximum_point_impulse_Ns=impulse,reported_world_time_s=row['reported_world_time_s'],row=row)
        minsep[pair]=min(minsep.get(pair,float('inf')),min([p['separation'] for p in row['points']],default=float('inf')))
        if all(p.startswith('/World/Robot/') for p in row['actors']):self_contacts.append(row)
    for pair,count in counts.most_common():contacts.append(dict(colliders=list(pair),reports=count,minimum_reported_separation_m=minsep[pair],**peaks[pair]))
    properties=json.loads((run/'resolved_shape_properties.json').read_text())
    offsets={Path(b['body']).name:(b['contact_offsets'][0],b['rest_offsets'][0]) for b in properties['bodies']}
    report=dict(scope='Post hoc source-hull/cooked-proxy/contact comparison; one diagnostic did not reproduce camera05 impulse, and contact-report overhead failed pace',self_contact_reports=len(self_contacts),cooked_shapes_checked=len(converted),source_hulls_contain_cooked_to_1um=bool(all(m['max_cooked_vertex_outside_raw_hull_m']<=1e-6 for m in metrics)),maximum_cooked_vertex_outside_source_hull_m=max(m['max_cooked_vertex_outside_raw_hull_m'] for m in metrics),shapes=metrics,contact_pairs=contacts,actual_forearm_contact_offset_m=offsets['ur5_forearm_link'][0],actual_wrist2_contact_offset_m=offsets['ur5_wrist_2_link'][0],duplicated_ground_contact_colliders=sorted({c for pair in counts for c in pair if c.startswith('/World/Ground/')}),known_scene_problem='Both the visual geom mesh and analytic collisionPlane produced real wheel contacts at the same ground height. GroundPlane creates the analytic plane, then its new-object branch additionally enables /geom collision. No robot collision pair should be removed.',causation_limit='The duplicate plane is directly observed; the exact cause of camera05 transient cannot be established from this run because the transient did not recur.')
    files=[Path(__file__),run/'contacts.jsonl',run/'resolved_shape_properties.json',run/'cooked_collision_representation.json',run/'sources/live_model.json',run/'acceptance.json']
    report['files_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['shapes','contact_pairs','files_sha256']},indent=2))


if __name__=='__main__':main()
