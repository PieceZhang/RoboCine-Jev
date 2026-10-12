"""Only validated, consecutively numbered combos belong to the active catalogue."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import generate_shot_trajectories as generator
from trajectory_record import resolve_recorded_trajectory
from current_combo_profile import REQUIRED_CONTROLLER_SOURCES, current_profile, recorded_dynamic_run


def test_historical_combo_catalog_is_consecutive_and_records_five_cm_evidence():
    catalogue = json.loads((ROOT/'trajectories/shot_validation.json').read_text())
    names = [f'shot_combo{i}.json' for i in range(1, 5)]
    assert sorted(p.name for p in (ROOT/'trajectories').glob('shot_combo[0-9].json')) == names
    assert sorted(p.name for p in (ROOT/'trajectories').glob('shot_combo*_actions.json')) == [
        f'shot_combo{i}_actions.json' for i in range(1, 5)]
    paired_names = [name for index in range(1, 5)
                    for name in (f'shot_combo{index}.json', f'shot_combo{index}_actions.json')]
    assert [row['trajectory'] for row in catalogue['results']] == paired_names
    assert catalogue['format'] == 'jvs_combo_pair_validation/v1'
    assert catalogue['accuracy_scope'] == 'shot_motion_intervals_only'
    assert catalogue['pass'] and catalogue['identical_controller_for_all_runs']
    # These are recorded acceptance claims for the October 2 controller and
    # trajectories. Current-profile applicability is checked separately.
    assert catalogue['accuracy_thresholds']['position_max_m'] == .05
    for pair in catalogue['pairs']:
        assert pair['similarity_pass'] and pair['both_accuracy_and_cinematic_pass']
        assert pair['identical_controller'] and pair['identical_scene'] and pair['identical_spawn']
        assert pair['position_rms_delta_m'] <= catalogue['similarity_thresholds']['position_rms_delta_m']
        assert pair['position_max_delta_m'] <= catalogue['similarity_thresholds']['position_max_delta_m']
    for row in catalogue['results']:
        folder = ROOT/row['result_directory']
        metadata = json.loads((folder/'run_metadata.json').read_text())
        summary = json.loads((folder/'tracking_summary.json').read_text())
        assert metadata['trajectory_sha256'] == row['trajectory_sha256']
        assert metadata['source_sha256'] == row['source_sha256']
        for metric in ('position_error_rms', 'position_error_max', 'orientation_error_rms'):
            assert summary[metric] == row[metric]
        assert row['completed'] and not row['failure']
        assert row['tracking_accuracy_pass'] and row['approximate_success']
        assert row['accuracy_scope'] == 'shot_motion_intervals_only'
        assert row['position_error_max'] <= .05
        assert row['position_error_rms'] <= .05
        assert row['orientation_error_rms'] <= .1
        assert row['all_phases_minimum_command_clearance_m'] >= .025
        assert row['source_sha256'] == catalogue['controller_source_sha256']
        assert not row['source_changed_during_run']


def test_current_profile_identifies_stale_historical_hashes_without_claiming_new_acceptance():
    profile = current_profile(ROOT)
    catalogue = json.loads((ROOT/'trajectories/shot_validation.json').read_text())
    assert profile['historical_catalogue_sha256'] == hashlib.sha256(
        (ROOT/'trajectories/shot_validation.json').read_bytes()).hexdigest()
    expected_changes = [path for path, digest in catalogue['controller_source_sha256'].items()
                        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest]
    assert profile['historical_controller_sources_changed'] == expected_changes
    for row in profile['trajectories']:
        digest = hashlib.sha256((ROOT/'trajectories'/row['trajectory']).read_bytes()).hexdigest()
        assert row['trajectory_sha256'] == digest
        assert row['historical_trajectory_matches_current'] == (digest == row['historical_trajectory_sha256'])
        assert row['pair_validation_status'] == 'not_revalidated_for_current_profile'
    assert not profile['paired_variants_revalidated']
    assert not profile['dynamic_demos_passed']
    assert all(row['validation_status'] == 'pending' for row in profile['dynamic_demos'].values())


def test_dynamic_profile_requires_explicit_matching_completed_run_and_rejects_failure(tmp_path):
    package = tmp_path/'package'
    trajectories = package/'trajectories'
    trajectories.mkdir(parents=True)
    trajectory = trajectories/'shot_combo1.json'
    trajectory.write_text('{}\n')
    folder = tmp_path/'run'
    folder.mkdir()
    (folder/'trajectory.json').write_bytes(trajectory.read_bytes())
    digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    hashes = {str(trajectory.resolve()): digest}
    for relative in REQUIRED_CONTROLLER_SOURCES:
        source = package/relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('captured controller source\n')
        hashes[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = dict(trajectory=str(trajectory.resolve()), source_sha256=hashes, sources_unchanged=True)
    (folder/'run.json').write_text(json.dumps(metadata))
    summary = dict(completed=True, failure='', tracking_accuracy_pass=True, approximate_success=True,
        accuracy_scope='shot_motion_intervals_only', execution_duration_s=45.,
        position_error_rms=.01, position_error_max=.02, orientation_error_rms=.01)
    (folder/'tracking_summary.json').write_text(json.dumps(summary))
    (folder/'telemetry.csv').write_text('wall_elapsed_s\n-3.0\n0.01\n45.0\n48.0\n')
    report = dict(dynamic_demo_pass=True, source_sha256=hashes, sources_unchanged=True)
    (folder/'dynamic_report.json').write_text(json.dumps(report))
    assert recorded_dynamic_run(package, 'crossing', folder)['validation_status'] == 'passed'
    # Completion before a minute is insufficient when the recorded stopping
    # tail extends the full physical-time video past a minute.
    (folder/'telemetry.csv').write_text('wall_elapsed_s\n0.01\n45.0\n60.01\n')
    result = recorded_dynamic_run(package, 'crossing', folder)
    assert result['validation_status'] == 'failed'
    assert not result['under_one_minute_including_stop']
    (folder/'telemetry.csv').write_text('wall_elapsed_s\n0.01\n45.0\n48.0\n')
    report['dynamic_demo_pass'] = False
    (folder/'dynamic_report.json').write_text(json.dumps(report))
    assert recorded_dynamic_run(package, 'crossing', folder)['validation_status'] == 'failed'
    report['dynamic_demo_pass'] = True
    (folder/'dynamic_report.json').write_text(json.dumps(report))
    summary.update(completed=False, failure='primal infeasible')
    (folder/'tracking_summary.json').write_text(json.dumps(summary))
    assert recorded_dynamic_run(package, 'crossing', folder)['validation_status'] == 'failed'
    trajectory.write_text('{"changed":true}\n')
    assert recorded_dynamic_run(package, 'crossing', folder)['validation_status'] == 'source_changed'


def test_manifest_and_obstacle_scene_reference_existing_numbered_combos():
    manifest = json.loads((ROOT/'trajectories/shot_stage2_manifest.json').read_text())
    combos = [row for row in manifest if row['name'].startswith('combo')]
    assert [row['name'] for row in combos] == [f'combo{i}' for i in range(1, 5)]
    for row in manifest:
        assert (ROOT/'trajectories'/row['trajectory']).is_file()
    scene = json.loads((ROOT/'scenes/combo.json').read_text())
    assert scene['trajectory'] == 'shot_combo1.json'


def test_regeneration_does_not_reintroduce_rejected_or_unnumbered_combos(tmp_path, monkeypatch):
    first = json.loads((ROOT/'trajectories/shot_combo1.json').read_text())
    target = first['scene_context']['target']['world_y_up_m']
    import numpy as np
    monkeypatch.setattr(generator, '_scene_target', lambda *_: np.array(target))
    manifest = generator.generate_validation_combos(tmp_path)
    assert len(manifest) == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == [f'shot_combo{i}.json' for i in range(1, 5)]
    for row in manifest:
        assert (tmp_path/row['trajectory']).read_bytes() == (ROOT/'trajectories'/row['trajectory']).read_bytes()


@pytest.mark.parametrize('mode', ['actions', 'both'])
def test_generator_produces_paired_variants_without_altering_full_trajectory(tmp_path, monkeypatch, mode):
    first = json.loads((ROOT/'trajectories/shot_combo1.json').read_text())
    import numpy as np
    monkeypatch.setattr(generator, '_scene_target', lambda *_: np.array(first['scene_context']['target']['world_y_up_m']))
    manifest = generator.generate_validation_combos(tmp_path, mode=mode)
    assert len(manifest) == (4 if mode == 'actions' else 8)
    for row in manifest:
        path = tmp_path/row['trajectory']
        assert path.read_bytes() == (ROOT/'trajectories'/path.name).read_bytes()


def test_external_result_path_is_supported(tmp_path):
    from analyze_combo_pairs import record_path
    assert record_path(tmp_path, ROOT) == str(tmp_path.resolve())


def test_historical_record_uses_hash_instead_of_reused_filename(tmp_path):
    trajectories = tmp_path/'trajectories'
    trajectories.mkdir()
    old_number = trajectories/'shot_combo1.json'
    renamed = trajectories/'shot_combo2.json'
    old_number.write_text('{"different": true}\n')
    renamed.write_text('{"validated": true}\n')
    metadata = dict(trajectory=str(old_number),
                    trajectory_sha256=hashlib.sha256(renamed.read_bytes()).hexdigest())
    assert resolve_recorded_trajectory(tmp_path, metadata, 'shot_combo1') == renamed
    metadata['trajectory_sha256'] = '0'*64
    with pytest.raises(FileNotFoundError, match='SHA-256'):
        resolve_recorded_trajectory(tmp_path, metadata, 'shot_combo1')


def test_original_negative_control_is_not_a_selectable_trajectory():
    fixture = ROOT/'test/fixtures/combo_unreachable.json'
    assert fixture.is_file()
    assert not (ROOT/'trajectories/shot_combo.json').exists()
    assert not (ROOT/'trajectories/shot_combo_reachable.json').exists()
    assert resolve_recorded_trajectory(ROOT, {}, 'shot_combo') == fixture
