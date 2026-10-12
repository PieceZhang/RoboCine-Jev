"""Measured-only triangle reconstruction; no Isaac or scene assets required."""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reconstruct_lidar_mesh import scan_mesh, weld_mesh, write_mesh, reconstruct
import jvs_isaac_sim as jvs
import jvs_simview_isaac as simview
import types


def wall_points(x=2., y_offset=0.):
    y, z = np.meshgrid(np.linspace(-.2, .2, 21) + y_offset, np.linspace(-.2, .2, 21))
    return np.column_stack((np.full(y.size, x), y.ravel(), z.ravel()))


def test_triangulates_real_returns_and_orients_toward_sensor(tmp_path):
    original = wall_points()
    vertices, faces = scan_mesh(original)
    assert len(faces) > 100
    assert set(map(tuple, vertices)) <= set(map(tuple, original))
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.einsum('ij,ij->i', normals, triangles.mean(axis=1)) < 0)
    vertices, faces = weld_mesh(vertices, faces)
    write_mesh(tmp_path, vertices, faces)
    assert f'element face {len(faces)}'.encode() in (tmp_path / 'scene_mesh.ply').read_bytes()[:400]
    assert sum(line.startswith('f ') for line in (tmp_path / 'scene_mesh.obj').read_text().splitlines()) == len(faces)


def test_does_not_bridge_large_unobserved_gaps():
    points = np.concatenate((wall_points(y_offset=-1.), wall_points(y_offset=1.)))
    vertices, faces = scan_mesh(points)
    assert len(faces)
    triangles = vertices[faces]
    assert np.all(np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2) <= .25)
    positive = triangles[:, :, 1] > 0
    assert np.all(positive == positive[:, :1])


def test_welding_removes_duplicate_and_degenerate_faces():
    vertices = np.array([[0., 0, 0], [.1, 0, 0], [0, .1, 0], [.001, 0, 0]])
    v, f = weld_mesh(vertices, np.array([[0, 1, 2], [3, 1, 2], [0, 3, 1]]))
    assert len(v) == 3 and len(f) == 1


def test_empty_or_invalid_mesh_is_rejected():
    with pytest.raises(ValueError, match='no supported triangles'):
        weld_mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    with pytest.raises(ValueError):
        scan_mesh(wall_points(), max_edge=-1)


def test_two_sensor_frames_fuse_in_world_and_do_not_fill_between(tmp_path):
    rows = []
    for name, y in [('rear', -1.), ('front', 1.)]:
        matrix = np.eye(4)
        matrix[3, 1] = y
        file = name + '.npz'
        np.savez(tmp_path / file, points=wall_points(), world_sensor=matrix,
                 world_base=np.eye(4), self_boxes=np.empty((0, 2, 3)),
                 self_box_frames=np.empty((0, 4, 4)))
        rows.append(dict(file=file, pose_index=0, sensor=name, stamp=1.0))
    (tmp_path / 'capture.json').write_text(json.dumps(dict(scans=rows, note='synthetic world poses')))
    output = tmp_path / 'mesh'
    vertices, faces = reconstruct(tmp_path, output)
    assert vertices[:, 1].min() < -.9 and vertices[:, 1].max() > .9
    assert (np.linalg.norm(vertices[faces] - np.roll(vertices[faces], 1, axis=1), axis=2) <= .25).all()
    report = json.loads((output / 'mesh_report.json').read_text())
    assert {r['sensor'] for r in report['used_scans']} == {'front', 'rear'}
    assert report['reference_geometry_used'] is False
    with pytest.raises(ValueError, match='fresh output'):
        reconstruct(tmp_path, output)


def test_dual_cli_defaults_and_duplicate_topic_validation():
    args = jvs.build_parser().parse_args(['--xacro', 'fixture'])
    assert args.lidar_mode == 'both'
    assert args.enable_front_lidar and args.front_lidar_config == 'Example_Rotary'
    assert jvs.enabled_lidars(args) == ('rear', 'front')
    jvs.validate_front_lidar_arguments(args)
    args.front_lidar_topic = 'lidar/points'
    with pytest.raises(ValueError, match='distinct'):
        jvs.validate_front_lidar_arguments(args)
    args.enable_lidar = False
    jvs.validate_front_lidar_arguments(args)
    assert jvs.enabled_lidars(args) == ()
    assert jvs.create_front_rtx_lidar(None, '', args) == ''


def test_front_only_mode_and_legacy_rear_only_mode():
    for parser, extra in ((jvs.build_parser(), []),
                          (simview.build_parser(), ['--simview-stage', 'fixture.usda'])):
        args = parser.parse_args(['--xacro', 'fixture', '--lidar-mode', 'front', *extra])
        args.lidar_config = ''  # A disabled rear sensor needs no valid profile.
        args.lidar_topic = ''
        args.lidar_frame_id = ''
        assert jvs.enabled_lidars(args) == ('front',)
        jvs.validate_front_lidar_arguments(args)
        args.front_lidar_topic = ''
        with pytest.raises(ValueError, match='front LiDAR config and topic'):
            jvs.validate_front_lidar_arguments(args)
        args.front_lidar_topic = '/lidar/front/points'
        args.enable_front_lidar = False
        with pytest.raises(ValueError, match='conflicts'):
            jvs.enabled_lidars(args)
        args.lidar_mode = 'both'
        assert jvs.enabled_lidars(args) == ('rear',)
        args.enable_lidar = False
        assert jvs.enabled_lidars(args) == ()


def test_ros_graph_has_distinct_publishers_and_shared_context(monkeypatch):
    edits = []
    keys = types.SimpleNamespace(CREATE_NODES='nodes', CONNECT='connections', SET_VALUES='values')
    core = types.ModuleType('omni.graph.core')
    core.Controller = types.SimpleNamespace(Keys=keys, edit=lambda *args: edits.append(args))
    omni, graph = types.ModuleType('omni'), types.ModuleType('omni.graph')
    omni.graph, graph.core = graph, core
    for name, value in [('omni', omni), ('omni.graph', graph), ('omni.graph.core', core),
                        ('usdrt', types.SimpleNamespace(Sdf=types.SimpleNamespace(Path=str))),
                        ('pxr', types.SimpleNamespace(Sdf=types.SimpleNamespace(Path=str)))]:
        monkeypatch.setitem(sys.modules, name, value)
    jvs.create_ros_graph('/robot/base', lidar_path='/robot/rear/sensor', front_lidar_path='/robot/front/sensor')
    edit = edits[0][1]
    names = [name for name, _ in edit['nodes']]
    assert len(names) == len(set(names))
    values = dict(edit['values'])
    assert values['PublishTF.inputs:parentPrim'] == ['/robot/base']
    assert values['PublishOdomTF.inputs:parentFrameId'] == 'odom'
    assert values['PublishOdomTF.inputs:childFrameId'] == 'base_link'
    assert values['PublishLidarPointCloud.inputs:frameId'] == 'lidar_link'
    assert values['FrontPublishLidarPointCloud.inputs:frameId'] == 'front_lidar_link'
    assert values['PublishLidarPointCloud.inputs:topicName'] == '/lidar/points'
    assert values['FrontPublishLidarPointCloud.inputs:topicName'] == '/lidar/front/points'
    assert values['FrontCreateLidarRenderProduct.inputs:cameraPrim'] == ['/robot/front/sensor']
    assert ('Context.outputs:context', 'FrontPublishLidarPointCloud.inputs:context') in edit['connections']
    edits.clear()
    jvs.create_ros_graph('/robot/base', front_lidar_path='/robot/front/sensor')
    front_only = edits[0][1]
    assert 'FrontPublishLidarPointCloud' in dict(front_only['nodes'])
    assert 'PublishLidarPointCloud' not in dict(front_only['nodes'])
    assert dict(front_only['values'])['FrontCreateLidarRenderProduct.inputs:cameraPrim'] == ['/robot/front/sensor']
    edits.clear()
    jvs.create_ros_graph('/robot/base')
    assert not any('Lidar' in name for name, _ in edits[0][1]['nodes'])
