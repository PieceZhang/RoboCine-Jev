from types import SimpleNamespace as NS
import numpy as np
import pytest

from jvs_mock.lidar_collision_map import LidarCollisionMap, cloud_xyz
from jvs_mock.kc_svsdf import SweptCollision


def test_cloud_row_padding_endian_and_invalid_returns():
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['>f4'] * 3,
                      'offsets': [0, 8, 12], 'itemsize': 16})
    data = bytearray(80)
    cloud = np.ndarray((2, 2), dtype=dtype, buffer=data, strides=(40, 16))
    cloud['x'] = [[1, 2], [3, np.nan]]
    cloud['y'] = 5
    cloud['z'] = 6
    msg = NS(fields=[NS(name=n, datatype=7, count=1, offset=o)
                    for n, o in zip('xyz', [0, 8, 12])], is_bigendian=True,
             height=2, width=2, row_step=40, point_step=16, data=data)
    np.testing.assert_equal(cloud_xyz(msg), [[1, 5, 6], [2, 5, 6], [3, 5, 6]])


def integrate(mapping, points, stamp=1., sensor='rear', translation=None):
    return mapping.integrate(points, np.eye(3), np.zeros(3) if translation is None else translation,
                             np.zeros(9), 0., np.eye(3), stamp, sensor)


def test_transform_self_floor_roi_filter_and_immutable_snapshot():
    mapping = LidarCollisionMap(confirmations=1)
    integrate(mapping, [[0, 0, .13], [1, 0, .01], [8, 0, .3], [1, 0, .5], [np.nan, 0, 0]])
    first = mapping.snapshot()
    assert len(first['surfel_centers']) == 1
    np.testing.assert_allclose(first['surfel_centers'][0], [1.0, .04, .52])
    integrate(mapping, [[1, 0, .5]], sensor='front', translation=[0, 1, 0])
    assert len(mapping.snapshot()['surfel_centers']) == 2
    assert len(first['surfel_centers']) == 1
    assert mapping.scans == {'rear': 1, 'front': 1}


def test_map_expiry_and_clearance_equivalence_to_measured_voxels():
    mapping = LidarCollisionMap(retention_s=2., confirmations=1)
    integrate(mapping, [[1, 0, .5]])
    before = mapping.snapshot()
    oracle = SweptCollision(before)
    distances = oracle.clearance(np.zeros(9))
    expected = oracle.geometry.distances(np.zeros(9), before['surfel_centers']) - np.sqrt(3)*.04
    np.testing.assert_allclose(distances, expected)
    integrate(mapping, [[2, 0, .5]], stamp=4.)
    assert len(mapping.cells) == 1
    assert mapping.snapshot()['surfel_centers'][0, 0] > 1.9
    assert oracle.centers[0, 0] == pytest.approx(1.)


@pytest.mark.parametrize('kwargs', [{'voxel_size': 0}, {'retention_s': -1},
    {'minimum_height': 2}, {'map_radius': np.nan}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        LidarCollisionMap(**kwargs)


def test_front_invalid_near_returns_do_not_create_obstacles():
    mapping = LidarCollisionMap(confirmations=1)
    mapping.integrate([[.5, 0, .3], [2, 0, .5]], np.eye(3), np.zeros(3),
        np.zeros(9), 0., np.eye(3), 1., 'front', minimum_range=1.)
    assert len(mapping.cells) == 1
    assert mapping.snapshot()['surfel_centers'][0, 0] > 1.9


def test_live_collision_replacement_is_used_by_execution_and_worker_snapshot():
    from jvs_mock.obstacle_predictive_qp import ObstaclePredictiveQP
    from jvs_mock.whole_body_qp import Limits
    mapping = LidarCollisionMap(confirmations=1)
    integrate(mapping, [[2, 0, .5]])
    controller = ObstaclePredictiveQP(Limits(), scene=mapping.snapshot())
    old = controller.collision.scene
    initial_clearance = controller.collision.clearance(np.zeros(9)).min()
    integrate(mapping, [[1, 0, .5]], stamp=2.)
    controller.update_collision_map(mapping.snapshot())
    assert controller.collision.clearance(np.zeros(9)).min() < initial_clearance
    assert len(old['surfel_centers']) == 1
    assert len(controller.collision.scene['surfel_centers']) == 2


def test_isolated_returns_need_independent_confirmation():
    mapping = LidarCollisionMap()
    integrate(mapping, [[1, 0, .5]], stamp=1.)
    assert not mapping.cells
    integrate(mapping, [[1, 0, .5]], stamp=1.01)
    assert not mapping.cells
    integrate(mapping, [[1, 0, .5]], stamp=1.1)
    assert len(mapping.cells) == 1
    integrate(mapping, [[2, 0, .5]], stamp=1.2)
    integrate(mapping, [[2, 0, .5]], stamp=3.)
    assert len(mapping.cells) == 1


def test_voxel_keys_match_numpy_row_unique_with_negative_coordinates():
    from jvs_mock.lidar_collision_map import voxel_keys
    points = np.random.default_rng(4).uniform(-3, 3, (5000, 3))
    keys, indices = voxel_keys(points, .08)
    expected, chosen = np.unique(np.floor(points / .08).astype(np.int64), axis=0, return_index=True)
    np.testing.assert_array_equal(keys, expected)
    np.testing.assert_array_equal(indices, chosen)


def test_empty_map_cannot_certify_a_command():
    mapping = LidarCollisionMap()
    with pytest.raises(RuntimeError, match='no observed surfaces'):
        SweptCollision(mapping.snapshot()).clearance(np.zeros(9))
