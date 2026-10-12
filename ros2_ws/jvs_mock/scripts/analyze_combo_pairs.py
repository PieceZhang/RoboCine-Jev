#!/usr/bin/env python3
"""Verify identical action geometry and compare paired actual Isaac executions.

Accuracy uses only declared motion intervals; safety/continuity still use the
whole execution. Similarity thresholds are explicit absolute differences,
not a change to the 5 cm peak gate. Failed trials remain in the results tree.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_shot_trajectories import motion_only_document


def record_path(path, package):
    try:
        return str(path.relative_to(package))
    except ValueError:
        return str(path.resolve())


def inspect(folder, package):
    metadata = json.loads((folder/'run_metadata.json').read_text())
    reference = json.loads((folder/'reference.json').read_text())
    summary = json.loads((folder/'tracking_summary.json').read_text())
    path = package/'trajectories'/f'{folder.name}.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['trajectory_sha256']:
        raise ValueError(f'{folder.name}: changed trajectory since validation')
    if hashlib.sha256(Path(metadata['stage']).read_bytes()).hexdigest() != metadata['stage_sha256']:
        raise ValueError(f'{folder.name}: changed scene since validation')
    with (folder/'telemetry.csv').open() as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r['phase'] == 'tracking']
    times = np.array([float(r['time_s']) for r in rows])
    active = np.zeros(len(rows), dtype=bool)
    shots = reference['execution_shots']
    for shot in shots:
        active |= (times >= shot.get('motion_start_s', shot['start_s'])) & (
            times <= shot.get('motion_end_s', shot['end_s']))
    if not np.any(active):
        raise ValueError(f'{folder.name}: no action samples')
    recorded = np.array([int(r['motion_active']) for r in rows], bool)
    if not np.array_equal(active, recorded):
        raise ValueError(f'{folder.name}: motion-mask telemetry mismatch')
    independent = {}
    for key in ('position_error', 'orientation_error'):
        values = np.array([float(r[key]) for r in rows])[active]
        independent[key+'_rms'] = float(np.sqrt(np.mean(values**2)))
        independent[key+'_max'] = float(np.max(values))
        for suffix in ('rms', 'max'):
            if not np.isclose(summary[key+'_'+suffix], independent[key+'_'+suffix], atol=1e-12):
                raise ValueError(f'{folder.name}: accuracy summary mismatch')
    independent['all_phases_position_error_max_m'] = max(float(r['position_error']) for r in all_rows)
    independent['all_phases_minimum_command_clearance_m'] = min(
        float(r['command_sweep_lower_m']) for r in all_rows)
    changed = [key for key, digest in metadata['source_sha256'].items()
               if hashlib.sha256((package/key).read_bytes()).hexdigest() != digest]
    record = dict(summary, name=folder.name, trajectory=path.name,
        trajectory_sha256=metadata['trajectory_sha256'],
        initial_base_pose=metadata['initial_base_pose'], result_directory=record_path(folder, package),
        scene=metadata['stage'], scene_sha256=metadata['stage_sha256'],
        source_sha256=metadata['source_sha256'],
        current_sources_match=not changed and not metadata.get('source_changed_during_run'),
        source_changed_during_run=metadata.get('source_changed_during_run', []),
        changed_sources=changed, execution_shots=shots, automatic_transitions=reference['automatic_transitions'],
        **independent)
    return record


def compare(full, actions, *, rms_delta=.01, peak_delta=.02):
    same_sources = full['source_sha256'] == actions['source_sha256']
    same_scene = (full.get('scene_sha256') is not None
                  and full['scene_sha256'] == actions.get('scene_sha256'))
    same_spawn = (full.get('initial_base_pose') is not None
                  and full['initial_base_pose'] == actions.get('initial_base_pose'))
    rms = abs(full['position_error_rms']-actions['position_error_rms'])
    peak = abs(full['position_error_max']-actions['position_error_max'])
    passed = all(r.get('completed') and not r.get('failure')
                 and r.get('accuracy_scope') == 'shot_motion_intervals_only'
                 and r.get('tracking_accuracy_pass') and r.get('approximate_success')
                 and r['position_error_max'] <= .05 and r['position_error_rms'] <= .05
                 and r['orientation_error_rms'] <= .1
                 and np.isfinite(r.get('all_phases_minimum_command_clearance_m', np.nan))
                 and r['all_phases_minimum_command_clearance_m'] >= .025
                 for r in (full, actions))
    return dict(full=full['trajectory'], actions=actions['trajectory'], identical_controller=same_sources,
        identical_scene=same_scene, identical_spawn=same_spawn,
        both_accuracy_and_cinematic_pass=bool(passed),
        position_rms_delta_m=rms, position_max_delta_m=peak,
        similarity_pass=bool(passed and same_sources and same_scene and same_spawn
                             and rms <= rms_delta and peak <= peak_delta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--rms-delta', type=float, default=.01)
    parser.add_argument('--peak-delta', type=float, default=.02)
    parser.add_argument('--write-trajectory-index', action='store_true')
    parser.add_argument('--retry-root', type=Path, action='append', default=[],
                        help='Explicit later trial directory; use its latest available record, including failures')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if any(not np.isfinite(v) or v < 0. for v in (args.rms_delta, args.peak_delta)):
        parser.error('similarity limits must be finite and nonnegative')
    package = Path(__file__).resolve().parents[1]
    root = args.results.resolve()
    trial_roots = [root, *(path.resolve() for path in args.retry_root)]
    def record_folder(name):
        candidates = [path/name for path in trial_roots if (path/name/'tracking_summary.json').exists()]
        if not candidates:
            raise FileNotFoundError(f'No execution record for {name}')
        return candidates[-1]
    reports, pairs = [], []
    for index in range(1, 5):
        name = f'shot_combo{index}'
        full_path = package/'trajectories'/f'{name}.json'
        action_path = package/'trajectories'/f'{name}_actions.json'
        if motion_only_document(json.loads(full_path.read_text())) != json.loads(action_path.read_text()):
            raise ValueError(f'{name}: action geometry does not match full trajectory')
        full = inspect(record_folder(name), package)
        actions = inspect(record_folder(name+'_actions'), package)
        reports.extend((full, actions))
        pairs.append(compare(full, actions, rms_delta=args.rms_delta, peak_delta=args.peak_delta))
    same_sources = all(r['source_sha256'] == reports[0]['source_sha256'] for r in reports)
    result = dict(format='jvs_combo_pair_validation/v1', catalog_date='2026-10-02',
        plant='Isaac Sim / PhysX', verification_scope='control_rate_samples_only',
        accuracy_scope='shot_motion_intervals_only',
        accuracy_thresholds=dict(position_rms_m=.05, position_max_m=.05, orientation_rms_rad=.1),
        similarity_thresholds=dict(position_rms_delta_m=args.rms_delta, position_max_delta_m=args.peak_delta),
        identical_controller_for_all_runs=same_sources,
        trial_roots=[record_path(path, package) for path in trial_roots],
        record_selection='latest explicitly listed trial, irrespective of accuracy; earlier raw trials retained',
        current_sources_match=all(r['current_sources_match'] for r in reports),
        controller_source_sha256=reports[0]['source_sha256'],
        pairs=pairs, results=reports)
    result['pass'] = bool(same_sources and result['current_sources_match'] and all(p['similarity_pass'] for p in pairs))
    (root/'pair_comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    if args.write_trajectory_index:
        if not result['pass']:
            raise ValueError('cannot replace active catalogue with a failed or changed-source validation')
        catalogue_path = package/'trajectories/shot_validation.json'
        old = json.loads(catalogue_path.read_text())
        result.update(scene=reports[0]['scene'], scene_sha256=reports[0]['scene_sha256'], excluded=old.get('excluded', []),
                      catalog_note='Four numbered combos, each with full and runtime-connected action-only variants. '
                                   '5 cm peak/RMS accuracy is evaluated only on action intervals. '
                                   'Safety/command continuity include every connector and hold.')
        catalogue_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(8, 5), layout='constrained')
        for suffix, label, offset in (('', 'Full', -.17), ('_actions', 'Actions + runtime connections', .17)):
            subset = [r for r in reports if r['name'].endswith('_actions') == bool(suffix)]
            x = np.arange(1, 5)+offset
            axes[0].bar(x, [r['position_error_max']*100 for r in subset], .34, label=label)
            axes[1].bar(x, [r['position_error_rms']*100 for r in subset], .34, label=label)
        axes[0].axhline(5., ls='--', color='black', lw=1, label='5 cm action peak limit')
        for ax, metric in zip(axes, ('peak', 'RMS')):
            ax.set(ylabel=f'Action position {metric} (cm)', xticks=range(1, 5))
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel('Combo catalogue number')
        figure.savefig(root/'pair_accuracy.png', dpi=170)
        plt.close(figure)
    print(json.dumps(dict(pass_all=result['pass'], pairs=pairs), indent=2))
    return result['pass']


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
