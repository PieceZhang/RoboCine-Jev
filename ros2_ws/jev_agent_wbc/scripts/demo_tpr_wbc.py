#!/usr/bin/env python3
"""Fixed-camera flying-obstacle closed-loop demo, with independent truth audit.

Only current obstacle position/velocity reach WBC; the flight function is used
by the environment and audit. No IK projection is applied to executed states.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_agent_wbc.planning import ExecutionWBC
from jev_agent_wbc.whole_body_qp import Limits, whole_body_kinematics
from jev_agent_wbc.actuation import advance_command
from jev_agent_wbc.kc_svsdf import CollisionGeometry
from jev_agent_wbc.jvs_kinematics import JOINT_LOWER, JOINT_UPPER

OBSTACLE_RADIUS = .07


class FixedCamera:
    shots = []
    sha256 = 'fixed-camera-six-dimensional-task'
    def __init__(self, state, duration):
        self.p, self.r, _ = whole_body_kinematics(state)
        self.duration = duration

    def sample(self, t):
        return self.p.copy(), self.r.copy(), np.zeros(6)


def flight(t, period=24.):
    angle = 2*np.pi*t/period+np.pi
    position = np.array([-.15+.12*np.cos(angle), .025*np.sin(2*angle), .43+.10*np.sin(angle)])
    velocity = 2*np.pi/period*np.array([-.12*np.sin(angle), .05*np.cos(2*angle), .10*np.cos(angle)])
    return position, velocity


def scene(t, period=24.):
    p, v = flight(t, period)
    return dict(format='jvs_lidar_surfels/v1', surfel_centers=[p.tolist()],
                surfel_radius=OBSTACLE_RADIUS, surfel_velocities=[v.tolist()], surfel_track_ids=[1],
                observation_stamp=t, minimum_clearance_m=.025,
                # Ideal current-state tracker, with an acceleration-derived
                # linear uncertainty envelope over the 1.5 s prediction window.
                # No future positions or flight phase are sent to the WBC.
                motion_uncertainty_rate_m_s=.012*(24./period)**2)


def run(args):
    from jev_agent_wbc.provenance import source_manifest
    args.output.mkdir(parents=True, exist_ok=False)
    manifest=source_manifest(Path(__file__).resolve().parents[1],
        {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    (args.output/'provenance.json').write_text(json.dumps(manifest,indent=2))
    state = np.array([0., 0., 0., 0., -.4, 1.7, 0., -1.3, 0.])
    initial = state.copy()
    trajectory = FixedCamera(state, args.duration)
    limits = Limits(command_ramp=True)
    controller = ExecutionWBC(limits, scene=scene(0., args.period))
    controller.set_trajectory(trajectory)
    if args.async_preview:
        from jev_agent_wbc.tpr_wbc import AsyncReconfigurationPreview
        controller.reconfiguration=AsyncReconfigurationPreview()
        controller.reconfiguration.reset(trajectory)
    controller.reconfiguration.period = args.preview_period
    geometry = CollisionGeometry()
    rows = []; failure = ''; start = time.perf_counter()
    for tick in range(round(args.duration/args.dt)):
        tick_start=time.perf_counter()
        t = tick*args.dt
        controller.update_collision_map(scene(t, args.period)); controller.set_time(t)
        old = controller.previous.copy()
        controller.measured_control = old.copy()
        try:
            u, info = controller.solve(state, *trajectory.sample(t), args.dt,
                                       trajectory_time=t, commanded_q=state[3:])
        except RuntimeError as error:
            failure = f't={t:.3f}: {error}'; break
        minimum = np.inf
        for sub in range(1, 6):
            now = t+sub*args.dt/5
            xsub = advance_command(state, old, old+sub/5*(u-old), sub/5*args.dt, True)
            obstacle, _ = flight(now, args.period)
            minimum = min(minimum, float(geometry.primitive_distances(xsub, obstacle[None])[0].min()-OBSTACLE_RADIUS))
        state = advance_command(state, old, u, args.dt, True)
        p, r, _ = whole_body_kinematics(state)
        obstacle, _ = flight(t+args.dt, args.period)
        per_link = geometry.distances(state, obstacle[None])[0]-OBSTACLE_RADIUS
        frozen = geometry.distances(initial, obstacle[None])[0]-OBSTACLE_RADIUS
        rows.append(dict(t=t+args.dt, state=state.tolist(), command=u.tolist(), obstacle=obstacle.tolist(),
                         camera_position=p.tolist(), camera_rotation=r.tolist(),
                         position_error_m=float(np.linalg.norm(p-trajectory.p)),
                         orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(r@trajectory.r.T).as_rotvec())),
                         clearance_m=minimum, per_link_clearance_m=per_link.tolist(),
                         frozen_clearance_m=float(frozen.min()),
                         velocity_ratio=float(np.max(np.abs(u)/limits.velocity)),
                         acceleration_ratio=float(np.max(np.abs(u-old)/(limits.acceleration*args.dt))),
                         wheel_speed_ratio=float(np.max(np.abs(limits.wheels@u))/limits.wheel_speed),
                         wheel_acceleration_ratio=float(np.max(np.abs(limits.wheels@(u-old)))/(limits.wheel_acceleration*args.dt)),
                         info=info))
        if tick % 250 == 0:
            print(f'{t:.1f}s: camera={rows[-1]["position_error_m"]*1000:.3f}mm clearance={minimum:.4f} status={info["tpr_status"]}', flush=True)
        if args.pace > 0:
            time.sleep(max(0.,args.dt/args.pace-(time.perf_counter()-tick_start)))
    controller.close()
    elapsed = time.perf_counter()-start
    summary = dict(backend='acceleration_limited_kinematic_closed_loop', duration_s=args.duration,
                   completed=len(rows)==round(args.duration/args.dt), failure=failure, wall_s=elapsed,
                   dt=args.dt, obstacle_period_s=args.period, frames=len(rows),
                   obstacle_radius_m=OBSTACLE_RADIUS,
                   received_camera_position=trajectory.p.tolist(), received_camera_rotation=trajectory.r.tolist(),
                   actual_state_integrated_without_pose_projection=True)
    summary['async_prediction']=args.async_preview
    if rows:
        states = np.array([initial]+[r['state'] for r in rows])
        summary.update(position_max_m=max(r['position_error_m'] for r in rows),
                       orientation_max_rad=max(r['orientation_error_rad'] for r in rows),
                       minimum_clearance_m=min(r['clearance_m'] for r in rows),
                       frozen_minimum_clearance_m=min(r['frozen_clearance_m'] for r in rows),
                       joint_excursion_rad=np.ptp(states[:, 3:], axis=0).tolist(),
                       base_xy_excursion_m=np.ptp(states[:, :2], axis=0).tolist(),
                       base_yaw_excursion_rad=float(np.ptp(states[:, 2])),
                       minimum_joint_margin_rad=float(np.minimum(states[:,3:]-JOINT_LOWER, JOINT_UPPER-states[:,3:]).min()),
                       degraded_steps=sum(r['info']['tpr_status']=='degraded' for r in rows),
                       warning_steps=sum(r['info']['tpr_status']=='warning' for r in rows),
                       prediction_valid_fraction=float(np.mean([r['info'].get('tpr_prediction_valid', False) for r in rows])),
                       solve_p50_p95_p99_s=np.percentile([r['info']['solve_s'] for r in rows], [50,95,99]).tolist(),
                       **{k:max(r[k] for r in rows) for k in ['velocity_ratio','acceleration_ratio','wheel_speed_ratio','wheel_acceleration_ratio']})
        summary['pass'] = (summary['completed'] and summary['minimum_clearance_m'] >= .025
                           and summary['position_max_m'] < .001 and summary['orientation_max_rad'] < .002
                           and summary['degraded_steps']==0
                           and all(summary[k] <= 1.+1e-6 for k in
                               ['velocity_ratio','acceleration_ratio','wheel_speed_ratio','wheel_acceleration_ratio']))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'telemetry.json').write_text(json.dumps(rows))
    print(json.dumps(summary, indent=2), flush=True)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=48.)
    parser.add_argument('--period', type=float, default=24.)
    parser.add_argument('--dt', type=float, default=.02)
    parser.add_argument('--preview-period', type=float, default=.4)
    parser.add_argument('--async-preview', action='store_true')
    parser.add_argument('--pace', type=float, default=0.)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
