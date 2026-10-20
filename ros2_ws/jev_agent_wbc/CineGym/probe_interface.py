"""Exercise real CPU/PhysX reset-step calls and save compact measured evidence.

The default tiny base-speed ramp is an interface probe, not an avoidance policy.
Use --stall-after to verify that the physical watchdog latches without pausing.
"""
import argparse
from pathlib import Path
import time
import numpy as np

from .env import CineGymEnv
from .episode import Episode, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'physx'), required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--isaac-python')
    parser.add_argument('--duration-limit', type=float)
    parser.add_argument('--stall-after', type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    options = dict(terminate_on_collision=False)
    if args.backend == 'physx':
        options.update(output=args.output, isaac_python=args.isaac_python, terminate_on_fault=False,
            extra_args=[] if args.duration_limit is None else ['--duration-limit', str(args.duration_limit)])
    rows, stalls = [], []
    with CineGymEnv(backend=args.backend, **options) as env:
        obs, reset = env.reset(episode=Episode.load(args.case))
        print('reset '+str({k:reset[k] for k in ('backend','episode_sha256')}), flush=True)
        first_t = obs['t']
        while True:
            if args.stall_after is not None and len(rows) == args.stall_after:
                before = time.monotonic()
                stalled_at = obs['t']
                time.sleep(.35)
                stalls.append(dict(before_sim_s=stalled_at, wait_wall_s=time.monotonic()-before))
            # A small commanded base motion proves commands reach the physical
            # articulation. Ramp from measured motion within the shared limits.
            action = np.zeros(8)
            action[0] = min(.01, max(0., float(obs['measured_control'][0]))+.002)
            if obs.get('control_status', {}).get('runtime_fault'):
                action[:] = 0.
            obs, reward, terminated, truncated, info = env.step(action)
            if stalls and 'after_sim_s' not in stalls[-1]:
                stalls[-1]['after_sim_s'] = obs['t']
            rows.append(dict(t=obs['t'], state=obs['state'].tolist(), camera_position=obs['camera_position'].tolist(),
                measured_control=obs['measured_control'].tolist(), reward=reward,
                sequence=obs.get('sequence'), source_sequence=info.get('submitted_action_sequence'),
                accepted_sequence=info.get('control_status', {}).get('accepted_action_sequence'),
                runtime_fault=info.get('runtime_fault'), evaluation=info.get('evaluation'),
                terminated=terminated, truncated=truncated, stream_error=info.get('stream_error')))
            if terminated or truncated:
                break
            if args.backend == 'cpu' and args.duration_limit is not None and obs['t']-first_t >= args.duration_limit:
                break
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'interaction.json', dict(backend=args.backend, reset=reset, rows=rows,
        probe_only=True, stall_after=args.stall_after, stalls=stalls,
        max_measured_speed_m_s=max(abs(r['measured_control'][0]) for r in rows),
        max_accepted_sequence=max((r['accepted_sequence'] or -1 for r in rows), default=-1),
        state_displacement_m=float(np.linalg.norm(np.array(rows[-1]['state'][:2])-np.array(rows[0]['state'][:2])))))
    print(f'saved {len(rows)} measured observations to {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
