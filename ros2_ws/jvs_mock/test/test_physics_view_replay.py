"""Recording provenance survives Kit's process-terminating shutdown."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import record_dynamic_physics_view as recording


def test_physics_provenance_is_saved_before_kit_exits(monkeypatch, tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    physics_file = source/'camera_physics.csv'
    physics_file.write_text('test rigid-body measurements')
    fingerprint = recording.renderer.digest(physics_file)
    (source/'camera_stability_physical.json').write_text(json.dumps(dict(
        camera_stability_pass=True, source_sha256={'camera_physics.csv': fingerprint})))
    run = SimpleNamespace(physics_file=physics_file, physics_sha256=fingerprint)
    monkeypatch.setattr(recording.renderer, 'load_run', lambda _: run)
    monkeypatch.setattr(recording, 'physical_replay', lambda value, _: value)

    def render(args, app, name, urdf):
        recording.renderer.load_run(source)
        return dict(demo=name, source_sha256={}, fps=30)

    def kit_main():
        record = recording.renderer.render_one(SimpleNamespace(output=tmp_path), None, 'two_movers', None)
        (tmp_path/'recordings.json').write_text(json.dumps([record]))
        raise SystemExit(0)

    monkeypatch.setattr(recording.renderer, 'render_one', render)
    monkeypatch.setattr(recording.renderer, 'main', kit_main)
    monkeypatch.setattr(recording.renderer, 'frame_annotation', recording.renderer.frame_annotation)
    monkeypatch.setattr(sys, 'argv', ['record_dynamic_physics_view.py', '--demo', 'two_movers'])
    with pytest.raises(SystemExit):
        recording.main()
    record = json.loads((tmp_path/'two_movers_recording.json').read_text())
    manifest = json.loads((tmp_path/'recordings.json').read_text())
    assert record['camera_stability_verified']
    assert record['pose_source_sha256'] == fingerprint
    assert not record['image_stabilization']
    assert manifest[0] == record
