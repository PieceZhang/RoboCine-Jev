#!/usr/bin/env python3
"""Separate static whole-body reachability from recorded execution error.

Constrained multistart IK supplies feasible witnesses, not a proof of global
infeasibility. Every returned witness is independently checked against all
surface voxels, the camera pose and joint limits. No controller is run here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jvs_mock.arm_comfort import COMFORT_POSE, preferred_posture
from jvs_mock.jvs_kinematics import JOINT_LOWER, JOINT_UPPER, JOINT_ORIGINS, CAMERA_OFFSET
from jvs_mock.kc_svsdf import SweptCollision
from jvs_mock.shot_predictive_qp import pose_jacobian
from jvs_mock.whole_body_qp import whole_body_kinematics
from jvs_mock.wbc_trajectory import CameraTrajectory


def trajectory_at_origin(path, reference=None):
    options = None
    if reference:
        document = json.loads(Path(reference).read_text())
        p = np.array(document['initial_camera_position'])
        r = np.array(document['initial_camera_rotation'])
        br = np.array(document['initial_base_rotation'])
        options = document.get('transition_options')
    else:
        p, r, _ = whole_body_kinematics(np.zeros(9), .002)
        br = np.eye(3)
    return CameraTrajectory(path, p, r, br, transition_options=options)


def inspect_record(folder, trajectory):
    rows = np.genfromtxt(Path(folder)/'telemetry.csv', delimiter=',', names=True,
                        dtype=None, encoding='utf-8')
    rows = rows[rows['phase'] == 'tracking']
    summary = []
    for shot in trajectory.shots:
        start = shot.get('motion_start_s', shot['start_s'])
        end = shot.get('motion_end_s', shot['end_s'])
        selected = rows[(rows['time_s'] >= start) & (rows['time_s'] <= end)]
        if not len(selected):
            continue
        e = selected['position_error']
        clock = rows['wall_elapsed_s'] if 'wall_elapsed_s' in rows.dtype.names else rows['time_s']
        intervals = np.maximum(np.r_[.02, np.diff(clock)], 1e-6)
        dt = intervals[(rows['time_s'] >= start) & (rows['time_s'] <= end)]
        worst = selected[np.argmax(e)]
        values = dict(type=shot['type'], samples=len(selected), accuracy_scope='shot_motion_intervals_only',
                      accuracy_start_s=start, accuracy_end_s=end,
                      rms_m=float(np.sqrt(np.mean(e**2))),
                      time_weighted_rms_m=float(np.sqrt(np.average(e**2, weights=dt))),
                      p95_m=float(np.percentile(e, 95)), max_m=float(np.max(e)),
                      worst_path_time_s=float(worst['time_s']))
        for name in ('clearance_m', 'arm_horizontal_reach_m', 'arm_joint_reserve_rad',
                     'heading_error', 'path_rate', 'controller_s', 'preview_age_s'):
            if name in rows.dtype.names:
                values['worst_'+name] = float(worst[name])
        for name in ('v', 'omega'):
            if 'command_'+name in rows.dtype.names:
                values['mean_abs_command_'+name] = float(np.mean(np.abs(selected['command_'+name])))
                values['mean_abs_measured_'+name] = float(np.mean(np.abs(selected['measured_'+name])))
        summary.append(values)
    return summary


def feasible_pose(collision, trajectory, t, previous=None, base_z=.002):
    tp, tr, _ = trajectory.sample(t)
    # The camera-to-base distance cannot exceed the entire arm length; this
    # local region contains every possible contact for the bounded IK search.
    chosen = collision.tree.query_ball_point(tp, 2.0) if collision.tree else np.arange(len(collision.centers))
    centers, radii = collision.centers[chosen], collision.radii[chosen]
    lower = np.r_[tp[:2]-.9, -3*np.pi, JOINT_LOWER+.01]
    upper = np.r_[tp[:2]+.9, 3*np.pi, JOINT_UPPER-.01]
    cache = {}

    def geometry(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            distances = collision.geometry.distances(x, centers, base_z)-radii[:, None]
            nearest = np.argmin(distances, axis=0)
            _, gradients = collision.geometry.distances(x, centers[nearest], base_z, gradient=True)
            cache.update(x=x.copy(), d=distances[nearest, np.arange(8)],
                         g=gradients[np.arange(8), np.arange(8)])
        return cache['d'], cache['g']

    def objective(x):
        p, r, C = pose_jacobian(x, base_z)
        error = np.r_[p-tp, Rotation.from_matrix(r@tr.T).as_rotvec()]
        weight = np.array([4000.]*3+[400.]*3)
        cost = .5*np.dot(error*weight, error)+1e-5*np.sum((x[3:]-COMFORT_POSE)**2)
        gradient = C.T@(weight*error)
        gradient[3:] += 2e-5*(x[3:]-COMFORT_POSE)
        return cost, gradient

    constraint = dict(type='ineq', fun=lambda x: geometry(x)[0]-collision.margin,
                      jac=lambda x: geometry(x)[1])
    tangent = trajectory.tangent(t)
    view = float(np.arctan2(tr[1, 0], tr[0, 0]))
    seeds = [] if previous is None else [previous]
    for yaw in (view, tangent if tangent is not None else view,
                (tangent if tangent is not None else view)+np.pi, view+np.pi/2, view-np.pi/2):
        q, offset = preferred_posture(yaw, tr, tp[2]-base_z)
        seeds.append(np.r_[tp[:2]-offset[:2], yaw, q])
    best = None
    started = time.monotonic()
    for seed in seeds:
        result = minimize(objective, np.clip(seed, lower, upper), jac=True, method='SLSQP',
                          bounds=list(zip(lower, upper)), constraints=[constraint],
                          options=dict(maxiter=150, ftol=1e-9))
        x = result.x
        p, r, _ = whole_body_kinematics(x, base_z)
        clearance = float(np.min(collision.geometry.distances(x, collision.centers, base_z)-collision.radii[:, None]))
        candidate = dict(time_s=float(t), position_error_m=float(np.linalg.norm(p-tp)),
                         orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(r@tr.T).as_rotvec())),
                         clearance_m=clearance, state=x.tolist(), optimizer_status=result.message,
                         objective=float(result.fun))
        candidate['pose_feasible'] = (candidate['position_error_m'] < .005
                                      and candidate['orientation_error_rad'] < .02
                                      and clearance >= collision.margin-1e-6)
        if best is None or (clearance >= collision.margin-1e-6, -candidate['objective']) > (
                best['clearance_m'] >= collision.margin-1e-6, -best['objective']):
            best = candidate
        if candidate['pose_feasible']:
            best = candidate
            break
    best['solve_s'] = time.monotonic()-started
    return best


def necessary_reach_bounds(collision, trajectory, times, base_z=.002, resolution=.01):
    """A necessary reach bound for the upright, static conservative model.

    The frustum contains a radius-0.13, height-0.12 cylinder. Exclude a grid
    cell only when an obstacle's inflated sphere covers its ENTIRE cell for
    that cylinder. All other cells, including uncertain boundary cells, stay
    free. Eight-neighbour connectivity is a superset of continuous free-space
    connectivity. This produces a lower bound on base-to-wrist distance.

    Remove the known orientation-dependent distal offsets. The joint2-to-
    joint5 chain cannot stretch beyond the sum of its link offset lengths.
    Minimize over every wrist-yaw sample and subtract its sampling bound.
    This is not an infeasibility certificate for an arbitrary tilted chassis
    or an uninflated triangle mesh; those are different collision models.
    """
    dz = np.maximum(np.abs(collision.centers[:, 2]-(base_z+.13))-.06, 0.)
    mask = dz < collision.radii+collision.margin
    centers = collision.centers[mask, :2]
    if not len(centers):
        return dict(scope='no obstacle intersects the inscribed base cylinder', bounds=[])
    radii = .13+np.sqrt((collision.radii[mask]+collision.margin)**2-dz[mask]**2)
    origin = np.minimum((centers-radii[:, None]).min(axis=0), [0., 0.])-1.
    high = np.maximum((centers+radii[:, None]).max(axis=0), [0., 0.])+1.
    shape = np.ceil((high-origin)/resolution).astype(int)+1
    cells = np.indices(shape).reshape(2, -1).T
    coords = origin+resolution*cells
    reserve = resolution/np.sqrt(2)
    distance, nearest = cKDTree(centers).query(coords, k=min(8, len(centers)))
    possible = ~np.any(distance+reserve < radii[nearest], axis=1)
    components, _ = label(possible.reshape(tuple(shape)), structure=np.ones((3, 3)))
    initial = np.rint(-origin/resolution).astype(int)
    initial_labels = components[initial[0]-1:initial[0]+2, initial[1]-1:initial[1]+2]
    reachable = np.isin(components.ravel(), np.unique(initial_labels[initial_labels > 0]))
    free_tree = cKDTree(coords[reachable])
    q6 = np.linspace(JOINT_LOWER[-1], JOINT_UPPER[-1], 721)
    tail = Rotation.from_euler('z', -q6).apply(JOINT_ORIGINS[5])+CAMERA_OFFSET
    chain = np.linalg.norm(JOINT_ORIGINS[2:5], axis=1).sum()
    yaw_sampling = np.linalg.norm(JOINT_ORIGINS[5, :2])*(q6[1]-q6[0])/2
    result = []
    for t in times:
        p, r, _ = trajectory.sample(t)
        wrist = p-tail@r.T
        horizontal = np.maximum(free_tree.query(wrist[:, :2])[0]-reserve-.098, 0.)
        vertical = wrist[:, 2]-base_z-.366
        bound = max(0., float(np.min(np.hypot(horizontal, vertical))-chain-yaw_sampling))
        result.append(dict(time_s=float(t), exact_orientation_position_error_lower_m=bound,
                           orientation_0p1rad_position_error_lower_m=max(0., bound-.1*(
                               np.linalg.norm(CAMERA_OFFSET)+np.linalg.norm(JOINT_ORIGINS[5])))))
    return dict(scope='upright base; conservative surfel spheres and 25 mm clearance',
                grid_resolution_m=resolution, uncertain_cells_retained=True,
                known_distal_offset_removed=True, bounds=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--times', type=float, nargs='+', help='Default: every second and the exact endpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--necessary-reach-bound', action='store_true')
    parser.add_argument('--triangle-vertices-only', action='store_true',
                        help='Optimistic diagnostic: actual mesh vertex points, zero radius; NOT a safety model')
    args = parser.parse_args()
    trajectory = trajectory_at_origin(args.trajectory, args.reference)
    if args.times is None:
        args.times = np.unique(np.r_[np.arange(0., trajectory.duration, 1.), trajectory.duration])
    collision = SweptCollision(args.mesh)
    report = dict(trajectory=str(args.trajectory.resolve()), mesh=str(args.mesh.resolve()),
                  scope='static multistart IK; feasible witnesses do not prove dynamic path feasibility', poses=[])
    if args.triangle_vertices_only:
        with np.load(args.mesh, allow_pickle=False) as mesh:
            vertices = np.asarray(mesh['vertices'], float)
        vertices = vertices[(vertices[:, 2] >= .03) & (vertices[:, 2] <= 1.6)]
        _, indices = np.unique(np.floor(vertices/.01).astype(int), axis=0, return_index=True)
        collision.centers = vertices[indices]
        collision.radii = np.zeros(len(indices))
        collision.tree = cKDTree(collision.centers)
        report['collision_scope'] = ('Optimistic subset of actual surface vertices, without voxel inflation. '
                                     'Feasible points do NOT certify triangle clearance or physical reachability.')
    if args.record:
        report['record'] = inspect_record(args.record, trajectory)
    if args.necessary_reach_bound:
        report['necessary_reach'] = necessary_reach_bounds(collision, trajectory, args.times)
        if args.triangle_vertices_only:
            report['necessary_reach']['scope'] = ('Upright base; optimistic subset of exact USD vertices, '
                                                   '25 mm clearance, no voxel inflation; not a full-mesh certificate')
        print(json.dumps(report['necessary_reach']), flush=True)
    previous = None
    for t in args.times:
        witness = feasible_pose(collision, trajectory, t, previous)
        report['poses'].append(witness)
        previous = np.array(witness['state']) if witness['pose_feasible'] else None
        print(json.dumps({k: v for k, v in witness.items() if k != 'state'}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
