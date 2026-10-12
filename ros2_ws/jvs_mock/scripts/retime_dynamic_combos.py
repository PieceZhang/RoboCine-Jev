#!/usr/bin/env python3
"""Speed up the three validated demos without changing any sampled pose.

The original files are archived beside the timing report, so repeated tuning
always starts from the validated full-distance trajectories. Camera holds at
inter-shot boundaries remain holds; the controller uses them for base arcs.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path


NAMES = ('shot_combo1', 'shot_combo3', 'shot_combo4')
COMBO_TIMING = {
    'shot_combo1': dict(factor=.5, connector_s=2., terminal_s=1.),
    'shot_combo3': dict(factor=.35, connector_s=2., terminal_s=1.,
                        motion_factors=(.35, 4.7/14., .5)),
    'shot_combo4': dict(factor=.5, connector_s=2., terminal_s=1.),
}


def retime_sampled(document, factor, connector_s, terminal_s, motion_factors=None):
    result = deepcopy(document)
    original_shots = document['shots']
    factors = ([factor]*len(original_shots) if motion_factors is None else list(motion_factors))
    if len(factors) != len(original_shots) or any(not 0 < value <= 1 for value in factors):
        raise ValueError('motion factors must provide one positive factor at most one per shot')
    knots = [(0., 0.)]
    cursor = float(original_shots[0]['motion_start_s'])
    knots.append((cursor, cursor))
    shots = []
    for index, shot in enumerate(original_shots):
        start = cursor
        end = start + factors[index] * (shot['motion_end_s'] - shot['motion_start_s'])
        item = deepcopy(shot)
        item['motion_start_s'] = start
        item['motion_end_s'] = end
        knots.append((float(shot['motion_end_s']), end))
        if index + 1 < len(original_shots):
            cursor = end + connector_s
            knots.append((float(original_shots[index + 1]['motion_start_s']), cursor))
        else:
            cursor = end + terminal_s
            knots.append((float(document['duration_s']), cursor))
        shots.append(item)
    previous_end = 0.
    for index, shot in enumerate(shots):
        shot['start_s'] = previous_end
        shot['end_s'] = (shot['motion_end_s'] + connector_s/2.
                         if index + 1 < len(shots) else cursor)
        previous_end = shot['end_s']
    def mapping(t):
        for (old_start, new_start), (old_end, new_end) in zip(knots, knots[1:]):
            if old_start <= t <= old_end:
                return new_start + (t-old_start)/(old_end-old_start)*(new_end-new_start)
        raise ValueError(f'timestamp outside original timeline: {t}')
    for sample in result['samples']:
        sample['time_from_start'] = round(mapping(sample['time_from_start']), 10)
    result['shots'] = shots
    result['duration_s'] = cursor
    result['sample_rate_hz'] = document['sample_rate_hz']/min(factors)
    result['retiming'] = dict(motion_time_factor=factor, inter_shot_window_s=connector_s,
        terminal_hold_s=terminal_s, original_duration_s=document['duration_s'],
        original_time_knots_s=[p[0] for p in knots], time_knots_s=[p[1] for p in knots],
        geometry_preserved='Every original pose sample is retained unchanged.',
        inter_shot_reference='stationary camera; controller base reconfiguration')
    if motion_factors is not None:
        result['retiming']['motion_time_factors_by_shot'] = factors
    assert [s['pose'] for s in result['samples']] == [s['pose'] for s in document['samples']]
    assert result['initial_base_pose'] == document['initial_base_pose']
    assert all(a['time_from_start'] < b['time_from_start']
               for a, b in zip(result['samples'], result['samples'][1:]))
    return result


def retime_actions(document, factor):
    result = deepcopy(document)
    for shot in result['shots']:
        shot['duration_s'] *= factor
        for sample in shot['samples']:
            sample['time_from_start'] = round(sample['time_from_start']*factor, 10)
    result['duration_s'] *= factor
    result['sample_rate_hz'] /= factor
    result['retiming'] = dict(motion_time_factor=factor,
        geometry_preserved='Every original pose sample is retained unchanged.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--names', nargs='+', choices=NAMES, default=NAMES)
    parser.add_argument('--motion-factor', type=float,
                        help='Override the configured per-combo motion time factor.')
    parser.add_argument('--shot-factors', type=float, nargs='+',
                        help='Override with one motion time factor per shot in each selected combo.')
    parser.add_argument('--connector-s', type=float)
    parser.add_argument('--terminal-s', type=float)
    args = parser.parse_args()
    if ((args.motion_factor is not None and not 0 < args.motion_factor <= 1)
            or any(value is not None and value <= 0 for value in (args.connector_s, args.terminal_s))
            or (args.shot_factors is not None and any(not 0 < value <= 1 for value in args.shot_factors))):
        parser.error('invalid positive timing values')
    original = args.report/'original_trajectories'
    original.mkdir(parents=True, exist_ok=True)
    report_path = args.report/'trajectory_timing.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in args.names:
        timing = dict(COMBO_TIMING[name])
        if args.motion_factor is not None:
            timing.pop('motion_factors', None)
        if args.shot_factors is not None:
            timing['motion_factors'] = args.shot_factors
        for parameter, override in (('factor', args.motion_factor),
                                    ('connector_s', args.connector_s), ('terminal_s', args.terminal_s)):
            if override is not None:
                timing[parameter] = override
        sampled = None
        for suffix, transform in (('', retime_sampled), ('_actions', retime_actions)):
            path = args.package/'trajectories'/f'{name}{suffix}.json'
            backup = original/path.name
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
            document = json.loads(backup.read_text())
            result = (transform(document, **timing)
                      if not suffix else transform(document, timing['factor']))
            if not suffix:
                sampled = result
            else:
                # Preserve the established exact full/actions pair contract.
                from generate_shot_trajectories import motion_only_document
                paired = motion_only_document(sampled)
                assert [[s['pose'] for s in shot['samples']] for shot in paired['shots']] == [
                    [s['pose'] for s in shot['samples']] for shot in document['shots']]
                result = paired
            path.write_text(json.dumps(result, indent=2)+'\n')
            report[path.name] = dict(original_duration_s=document['duration_s'],
                new_duration_s=result['duration_s'], retiming=result['retiming'],
                shots=[{k: v for k, v in shot.items() if k != 'samples'} for shot in result['shots']])
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: data['new_duration_s'] for name, data in report.items()}))


if __name__ == '__main__':
    main()
