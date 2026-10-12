#!/usr/bin/env python3
"""Summarize retained Isaac runs without excluding peaks or failed controls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnose_combo import trajectory_at_origin, inspect_record
from trajectory_record import resolve_recorded_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    package = Path(__file__).resolve().parents[1]
    runs = []
    for path in sorted(root.rglob('tracking_summary.json')):
        folder = path.parent
        summary = json.loads(path.read_text())
        metadata_path = folder/'run_metadata.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        name = metadata.get('trajectory', '')
        recorded_name = Path(name).stem if name else ('shot_combo' if folder.name == 'baseline' else folder.name)
        trajectory_path = resolve_recorded_trajectory(package, metadata, recorded_name, folder)
        trajectory = trajectory_at_origin(trajectory_path, folder/'reference.json')
        segments = inspect_record(folder, trajectory)
        (folder/'segment_diagnostics.json').write_text(json.dumps(segments, indent=2)+'\n')
        rows = np.genfromtxt(folder/'telemetry.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
        all_peak = float(np.max(rows['position_error']))
        tracking = rows[rows['phase'] == 'tracking']
        intervals = (tracking['dt_s'] if 'dt_s' in tracking.dtype.names else
                     np.maximum(np.r_[.02, np.diff(tracking['wall_elapsed_s'])], 1e-6))
        weighted_rms = float(np.sqrt(np.average(tracking['position_error']**2, weights=intervals)))
        retained_hashes_match = all(hashlib.sha256((package/key).read_bytes()).hexdigest() == value
                                   for key, value in metadata.get('source_sha256', {}).items())
        runs.append(dict(run=str(folder.relative_to(root)), trajectory=trajectory_path.name,
            recorded_trajectory=recorded_name+'.json',
            completed=summary['completed'], failure=summary['failure'],
            position_rms_m=summary.get('position_error_rms'),
            position_time_weighted_rms_m=weighted_rms,
            position_p95_m=float(np.quantile(tracking['position_error'], .95)),
            position_peak_m=summary.get('position_error_max'), all_phases_peak_m=all_peak,
            orientation_rms_rad=summary.get('orientation_error_rms'),
            minimum_command_clearance_m=summary.get('minimum_command_clearance_m'),
            minimum_observed_model_clearance_m=float(np.min(rows['clearance_m'])),
            controller_p95_ms=1000*summary.get('controller_p95_s', 0.),
            peak_5cm_pass=bool(summary['completed'] and not summary['failure']
                              and all_peak <= .05 and summary.get('position_error_rms', np.inf) <= .05),
            tracking_accuracy_pass=summary.get('tracking_accuracy_pass'),
            approximate_success=summary.get('approximate_success'),
            current_sources_match=retained_hashes_match if metadata else None,
            segments=segments))
    (root/'comparison.json').write_text(json.dumps(runs, indent=2)+'\n')
    comparisons = [('baseline', 'Original path, original controller', '#c74d45'),
                   ('reachable_baseline/shot_combo_reachable', 'New path, original controller', '#ba8a29'),
                   ('separate_execution_priority/shot_combo_reachable', 'New path, optimized controller', '#23785b')]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    for name, label, color in comparisons:
        folder = root/name
        if not (folder/'telemetry.csv').exists():
            continue
        rows = np.genfromtxt(folder/'telemetry.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
        rows = rows[rows['phase'] == 'tracking']
        axes[0].plot(rows['time_s'], rows['position_error']*100, label=label, color=color, linewidth=1.3)
        axes[1].plot(rows['time_s'], rows['clearance_m']*100, label=label, color=color, linewidth=1.3)
    axes[0].axhline(5., color='black', linestyle='--', linewidth=1., label='5 cm peak target')
    axes[1].axhline(2.5, color='black', linestyle='--', linewidth=1., label='2.5 cm clearance requirement')
    axes[0].set_ylabel('Camera position error (cm)')
    axes[1].set_ylabel('Model clearance at measured state (cm)')
    axes[1].set_xlabel('Trajectory time (s); physical duration is reported separately')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', fontsize=8)
    fig.savefig(root/'tracking_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps([{k: v for k, v in run.items() if k != 'segments'} for run in runs], indent=2))


if __name__ == '__main__':
    main()
