#!/usr/bin/env python3
"""Validate and publish a completed demo with simultaneous solid-collision evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PACKAGE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(staging, demo):
    recording = json.loads((staging / f'{demo}_recording.json').read_text())
    source = Path(recording['source_run']).resolve()
    report = json.loads((source / 'dynamic_report.json').read_text())
    collision = json.loads((source / 'counterfactual_collision.json').read_text())
    metadata = json.loads((source / 'run.json').read_text())
    assert report['completed'] and not report['failure'], 'incomplete closed-loop run'
    assert report['motion_timing_pass'], 'active shots must retain their fixed timing'
    assert report['sources_unchanged'] and all(digest(p) == h for p, h in metadata['source_sha256'].items())
    assert collision['baseline_completed'] and collision['simultaneous_collision_pass']
    assert collision['same_controller_sources'] and collision['same_camera_trajectory']
    assert collision['scene_sha256'] == digest(source / 'obstacle_scene.json')
    assert collision['baseline_telemetry_sha256'] == digest(Path(collision['baseline_directory']) / 'telemetry.csv')
    if demo == 'crossing':
        assert collision['arm_posture_changed_during_conflict'], 'arm must actually adjust during the conflict'
    assert report['physics_contact_audit_enabled'] and report['physics_contact_coverage_complete']
    for key in ['robot_obstacle_contact_events', 'robot_static_contact_events', 'obstacle_static_contact_events']:
        assert report[key] == 0, key
    # Relaxed presentation limits do not change any strict analysis gates.
    assert report['position_rms_cm'] <= 15. and report['position_peak_cm'] <= 40., 'tracking unsuitable for a relaxed demo'
    assert report['orientation_rms_rad'] <= .2, 'orientation tracking unsuitable for a demo'
    assert recording['source_files_unchanged'] and recording['source_usd_assets_unchanged']
    assert recording['maximum_camera_fk_interpolation_error_m'] <= .001
    for name, expected in recording['source_sha256'].items():
        assert digest(source / name) == expected
    annotations = json.loads((staging / recording['frame_annotations']).read_text())
    assert len(annotations) == recording['frames']
    for frame in annotations:
        assert frame['sample_wall_elapsed_s'] is None or frame['sample_wall_elapsed_s'] <= frame['physical_time_s'] + 1e-9
        assert (frame['position_error_cm'] is not None) == frame['recording']
    video = staging / recording['video']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(video)]))
    stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    assert [stream['width'], stream['height']] == recording['resolution']
    assert stream['r_frame_rate'] == f"{recording['fps']}/1"
    assert int(stream['nb_frames']) == recording['frames']
    assert abs(float(stream['duration']) - recording['frames'] / recording['fps']) < .001
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video), '-f', 'null', '-'], check=True)
    validation = dict(demo=demo, video=video.name, source_run=str(source),
        release_policy='user_authorized_relaxed_acceptance',
        relaxed_presentation_limits=dict(position_rms_cm=15., position_peak_cm=40., orientation_rms_rad=.2),
        strict_dynamic_demo_pass=report['dynamic_demo_pass'],
        fixed_motion_timing_pass=report['motion_timing_pass'],
        simultaneous_uninflated_solid_collision_without_avoidance=True,
        collision_evidence_scope='Offline measured-baseline geometric counterfactual; no deliberate physical contact trial.',
        counterfactual_collision=collision,
        position_rms_cm=report['position_rms_cm'], position_peak_cm=report['position_peak_cm'],
        orientation_rms_rad=report['orientation_rms_rad'],
        minimum_sampled_dynamic_clearance_lower_m=report['minimum_sampled_dynamic_clearance_lower_m'],
        physics_constraints_pass=report['physics_constraints_pass'],
        robot_obstacle_contact_events=report['robot_obstacle_contact_events'],
        robot_static_contact_events=report['robot_static_contact_events'],
        obstacle_static_contact_events=report['obstacle_static_contact_events'],
        source_files_unchanged=True, current_controller_sources_match=True,
        source_usd_assets_unchanged=True, media_decode_pass=True,
        layout=recording['layout'], view_count=recording['view_count'],
        dimensions=recording['resolution'], fps=recording['fps'], frames=recording['frames'],
        duration_s=float(stream['duration']), video_sha256=digest(video))
    validation_name = f'{demo}_video_validation.json'
    (staging / validation_name).write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    proof_name = f'{demo}_counterfactual_collision.json'
    (staging / proof_name).write_text(json.dumps(collision, ensure_ascii=False, indent=2) + '\n')
    for target in [PACKAGE / 'results/dynamic_dual_view_20261008_1148', PACKAGE / 'results/dynamic_interaction_20261008/videos']:
        target.mkdir(parents=True, exist_ok=True)
        names = [video.name, f'{demo}_recording.json', recording['frame_annotations'], validation_name, proof_name]
        for name in names:
            temporary = target / (name + '.replacement')
            shutil.copy2(staging / name, temporary)
            temporary.replace(target / name)
        if recording['layout'] == 'quad':
            legacy = f'{demo}_{recording["combo"]}_dual_view.mp4'
            temporary = target / (legacy + '.replacement')
            shutil.copy2(video, temporary)
            temporary.replace(target / legacy)
        (target / 'previews').mkdir(exist_ok=True)
        for preview in (staging / 'previews').glob(f'{demo}_*.png'):
            shutil.copy2(preview, target / 'previews' / preview.name)
        manifest = target / 'recordings.json'
        existing = json.loads(manifest.read_text()) if manifest.exists() else []
        combined = {item['demo']: item for item in existing}
        combined[demo] = recording
        manifest.write_text(json.dumps(list(combined.values()), ensure_ascii=False, indent=2) + '\n')
        print(target / video.name, flush=True)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--demo', choices=['crossing', 'overtake'], required=True)
    args = parser.parse_args()
    publish(args.staging.resolve(), args.demo)


if __name__ == '__main__':
    main()
