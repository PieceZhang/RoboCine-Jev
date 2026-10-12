#!/usr/bin/env python3
"""Report current hashes without reusing historical pair acceptance claims.

Only explicitly supplied dynamic runs can validate the three current demos.
The original pair catalogue and its recorded hashes remain historical evidence.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DEMOS = dict(crossing='shot_combo1.json', overtake='shot_combo3.json', two_movers='shot_combo4.json')
REQUIRED_CONTROLLER_SOURCES = (
    'jvs_mock/dynamic_whole_body_controller.py', 'jvs_mock/whole_body_controller.py',
    'jvs_mock/dynamic_predictive_qp.py', 'jvs_mock/obstacle_predictive_qp.py',
    'config/whole_body_controller.yaml', 'config/shot_whole_body_controller.yaml')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_dynamic_run(package, demo, folder):
    folder = Path(folder).resolve()
    result = dict(result_directory=str(folder), validation_status='pending')
    if not (folder/'tracking_summary.json').is_file() or not (folder/'run.json').is_file():
        return result
    metadata = json.loads((folder/'run.json').read_text())
    summary = json.loads((folder/'tracking_summary.json').read_text())
    report_path = folder/'dynamic_report.json'
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    telemetry_path = folder/'telemetry.csv'
    elapsed = []
    if telemetry_path.is_file():
        with telemetry_path.open() as source:
            elapsed = [float(row['wall_elapsed_s']) for row in csv.DictReader(source)]
    # Negative samples are the measured warm-up before playback starts at zero.
    valid_clock = bool(elapsed and elapsed[-1] >= 0. and all(math.isfinite(t) for t in elapsed)
                       and all(b > a for a, b in zip(elapsed, elapsed[1:])))
    physical_duration = elapsed[-1] if valid_clock else None
    # Match the recorder's inclusive physical timestamps at 30 fps. The stop
    # tail is part of the delivered video and cannot be hidden by camera-clock
    # timing or the earlier tracking-completion timestamp.
    video_duration = ((math.ceil(physical_duration * 30.) + 1) / 30.
                      if physical_duration is not None else None)
    expected_trajectory = package/'trajectories'/DEMOS[demo]
    recorded = metadata.get('source_sha256', {})
    changed = [name for name, digest in recorded.items()
               if not Path(name).is_file() or sha256(Path(name)) != digest]
    trajectory_digest = recorded.get(str(expected_trajectory.resolve()))
    captured = folder/'trajectory.json'
    has_controller_evidence = all(str((package/path).resolve()) in recorded
                                  for path in REQUIRED_CONTROLLER_SOURCES)
    matches = bool(recorded and not changed and metadata.get('sources_unchanged') is True
                   and has_controller_evidence
                   and Path(metadata.get('trajectory', '')).resolve() == expected_trajectory.resolve()
                   and trajectory_digest == sha256(expected_trajectory)
                   and captured.is_file() and sha256(captured) == trajectory_digest)
    audit_passed = bool(report.get('dynamic_demo_pass') is True
                        and report.get('source_sha256') == recorded
                        and report.get('sources_unchanged') is True)
    duration_passed = bool(video_duration is not None and video_duration < 60.)
    passed = bool(matches and audit_passed and duration_passed
                  and summary.get('completed') and not summary.get('failure')
                  and summary.get('tracking_accuracy_pass') and summary.get('approximate_success')
                  and summary.get('accuracy_scope') == 'shot_motion_intervals_only'
                  and summary.get('execution_duration_s', float('inf')) < 60.
                  and summary.get('position_error_rms', float('inf')) <= .02
                  and summary.get('position_error_max', float('inf')) <= .05
                  and summary.get('orientation_error_rms', float('inf')) <= .1)
    result.update(validation_status='passed' if passed else 'source_changed' if not matches else 'failed',
        source_sha256=recorded, trajectory_sha256=trajectory_digest,
        current_sources_match=matches, changed_sources=changed,
        controller_sources_recorded=has_controller_evidence,
        verification_scope=summary.get('verification_scope'),
        accuracy_scope=summary.get('accuracy_scope'),
        completed=summary.get('completed'), failure=summary.get('failure'),
        execution_duration_s=summary.get('execution_duration_s'),
        physical_duration_including_stop_s=physical_duration,
        expected_30fps_video_duration_s=video_duration,
        under_one_minute_including_stop=duration_passed,
        independent_dynamic_audit_passed=audit_passed,
        independent_dynamic_report_sha256=sha256(report_path) if report_path.is_file() else None,
        position_error_rms_m=summary.get('position_error_rms'),
        position_error_max_m=summary.get('position_error_max'),
        orientation_error_rms_rad=summary.get('orientation_error_rms'))
    return result


def current_profile(package, runs=None):
    package = Path(package).resolve()
    historical_path = package/'trajectories/shot_validation.json'
    historical = json.loads(historical_path.read_text())
    historical_sources = historical['controller_source_sha256']
    current_sources = {relative: sha256(package/relative) for relative in historical_sources}
    current_sources.update({str(path.relative_to(package)): sha256(path)
                            for path in sorted((package/'jvs_mock').glob('dynamic_*.py'))})
    changed_sources = [name for name, digest in historical_sources.items()
                       if current_sources[name] != digest]
    rows = []
    for recorded in historical['results']:
        path = package/'trajectories'/recorded['trajectory']
        digest = sha256(path)
        row = dict(trajectory=path.name, trajectory_sha256=digest,
            historical_result_directory=recorded['result_directory'],
            historical_trajectory_sha256=recorded['trajectory_sha256'],
            historical_trajectory_matches_current=digest == recorded['trajectory_sha256'],
            historical_controller_matches_current=not changed_sources,
            pair_validation_status='not_revalidated_for_current_profile',
            dynamic_validation=None)
        for demo, filename in DEMOS.items():
            if filename == path.name:
                row['dynamic_validation'] = (recorded_dynamic_run(package, demo, runs[demo])
                    if runs and demo in runs else dict(validation_status='pending'))
        rows.append(row)
    demos = {demo: next(row['dynamic_validation'] for row in rows if row['trajectory'] == filename)
             for demo, filename in DEMOS.items()}
    return dict(format='jvs_current_combo_profile/v1',
        scope='three_dynamic_demo_full_trajectories',
        controller_source_sha256=current_sources, trajectories=rows,
        historical_catalogue=str(historical_path.relative_to(package)),
        historical_catalogue_sha256=sha256(historical_path),
        historical_controller_sources_changed=changed_sources,
        dynamic_demos=demos,
        dynamic_demos_passed=all(row['validation_status'] == 'passed' for row in demos.values()),
        paired_variants_revalidated=False,
        evidence_note='Historical full/action pairs retain their recorded hashes. '
                      'Only explicitly supplied completed runs can validate the current dynamic profile; '
                      'combo2 and action-only variants have no new pair acceptance claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='append', default=[], metavar='DEMO=FOLDER',
                        help='Explicit current run; repeated demo uses the latest argument, even if failed')
    args = parser.parse_args()
    runs = {}
    for item in args.run:
        demo, separator, folder = item.partition('=')
        if not separator or demo not in DEMOS:
            parser.error('--run expects crossing, overtake or two_movers followed by =FOLDER')
        runs[demo] = Path(folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = current_profile(args.package, runs)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), dynamic_demos_passed=result['dynamic_demos_passed'],
                         status={demo: row['validation_status'] for demo, row in result['dynamic_demos'].items()})))


if __name__ == '__main__':
    main()
