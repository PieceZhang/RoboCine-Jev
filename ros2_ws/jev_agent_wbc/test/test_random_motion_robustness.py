"""Causal sensing/replanning regressions exposed by random moving obstacles."""
from concurrent.futures import Future
import numpy as np
import pytest
from scipy.spatial import cKDTree

from jev_agent_wbc.dynamic_lidar_map import DynamicLidarMap
from jev_agent_wbc.tpr_wbc import (
    AsyncReconfigurationPreview, ReconfigurationPreview, motion_evidence, motion_changed)
from test_tpr_wbc import STATE, Reference, controller, scene


def surface(center, radius=.1):
    az, el = np.meshgrid(np.linspace(-.7, .7, 15), np.linspace(-.6, .6, 11))
    return np.asarray(center) + radius*np.c_[
        (np.cos(el)*np.cos(az)).ravel(), (np.cos(el)*np.sin(az)).ravel(), np.sin(el).ravel()]


def test_stationary_history_does_not_veto_rigid_object_starting_to_move():
    mapping = DynamicLidarMap(voxel_size=.035, cluster_distance=.08, velocity_window_s=.8)
    initial = np.array([.8, .3, .6])
    cloud = surface(initial)
    mapping.static_world_tree = cKDTree(cloud)
    for t in np.arange(0., 1.21, .1):
        # Static evidence deliberately persists during the onset of motion.
        center = initial + [0., .15*max(0., t-.5)**2, 0.]
        mapping.current_points = surface(center)
        mapping.track(mapping.current_points, float(t+1.))
    assert len(mapping.tracks) == 1
    track = next(iter(mapping.tracks.values()))
    assert track['moving']
    assert track['static_fraction'] > .25
    np.testing.assert_allclose(track['velocity'], [0., .21, 0.], atol=1e-8)


def test_static_sphere_stays_static_when_surface_coverage_changes():
    mapping = DynamicLidarMap(voxel_size=.035, cluster_distance=.08)
    cloud = surface([.8, .3, .6])
    mapping.static_world_tree = cKDTree(cloud)
    for i in range(20):
        packet = cloud if i % 2 else cloud[cloud[:, 1] > .29]
        mapping.current_points = packet
        mapping.track(packet, 1.+.1*i)
    assert all(not tr['moving'] for tr in mapping.tracks.values())


def test_adjacent_cluster_does_not_contaminate_round_shape_fit():
    mapping = DynamicLidarMap(voxel_size=.035, cluster_distance=.08)
    first, second = surface([.8, .3, .2]), surface([.8, .3, .44])
    mapping.current_points = np.vstack([first, second])
    mapping.track(mapping.current_points, 1.)
    assert len(mapping.tracks) == 2
    for track in mapping.tracks.values():
        assert track.get('sphere') is not None
        assert abs(track['sphere']['radius']-.1) < 1e-8


def test_confirmed_mover_reassociates_after_leaving_local_prediction_gate():
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08)
    first=np.array([.8,.3,.6])
    mapping.current_points=surface(first)
    mapping.track(mapping.current_points,1.)
    identity=next(iter(mapping.tracks));track=mapping.tracks[identity]
    track.update(moving=True,velocity=np.array([.3,0.,0.]),
        model_extent=track['extent'].copy(),model_center_z=first[2])
    # After a brief occlusion, observed reversal puts the complete surface
    # outside the narrow predicted box but within the global association gate.
    actual=first+[-.12,0.,0.]
    mapping.current_points=surface(actual)
    mapping.track(mapping.current_points,1.5)
    assert list(mapping.tracks)==[identity]
    np.testing.assert_allclose(mapping.tracks[identity]['center'],actual,atol=1e-8)
    assert mapping.tracks[identity]['stamp']==1.5






def test_unconfirmed_rigid_object_retains_solid_occupancy():
    mapping = DynamicLidarMap(voxel_size=.035, cluster_distance=.08)
    center=np.array([.8,.3,.6])
    mapping.current_points=surface(center)
    mapping.track(mapping.current_points,1.)
    mapping.stamp=1.
    snapshot=mapping.snapshot()
    assert len(snapshot['tracks'])==1
    assert not snapshot['tracks'][0]['moving_confirmed']
    np.testing.assert_allclose(snapshot['tracks'][0]['velocity'],np.zeros(3))
    ids=np.flatnonzero(snapshot['surfel_track_ids']>0)
    assert len(ids)==1
    np.testing.assert_allclose(snapshot['surfel_centers'][ids[0]],center,atol=1e-8)
    assert snapshot['surfel_radii'][ids[0]]>=.1+snapshot['surfel_radius']-1e-8


def test_fused_historical_surfaces_do_not_refresh_or_spawn_tracks(monkeypatch):
    from collections import deque
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08)
    cloud=surface([-.8,.3,.6])
    mapping.current_points=cloud
    mapping.track(cloud,1.)
    identifier=next(iter(mapping.tracks))
    mapping.latest['front']=deque([(1.,cloud)])
    mapping.stamp=1.
    scan=dict(points=np.empty((0,3)),rotation=np.eye(3),translation=np.zeros(3),
        state=STATE.copy(),base_z=.002,base_rotation=np.eye(3),stamp=1.1,sensor='front',acquisition_duration_s=0.)
    mapping.integrate_scans([scan])
    assert list(mapping.tracks)==[identifier]
    assert mapping.tracks[identifier]['stamp']==1.
    assert mapping.snapshot()['tracks'][0]['age_s']==pytest.approx(.1)


def test_occluded_turning_sphere_is_enclosed_without_future_observations():
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08)
    center=np.array([.8,.3,.6])
    mapping.current_points=surface(center)
    mapping.track(mapping.current_points,1.)
    track=next(iter(mapping.tracks.values()))
    track.update(moving=True,model_extent=track['extent'],velocity=np.array([.2,0.,0.]))
    mapping.stamp=1.4
    snapshot=mapping.snapshot()
    ids=np.flatnonzero(snapshot['surfel_track_ids']>0)
    predicted=center+track['velocity']*.4
    np.testing.assert_allclose(snapshot['surfel_centers'][ids[0]],predicted,atol=1e-8)
    radius=snapshot['surfel_radii'][ids[0]]
    assert snapshot['tracks'][0]['occlusion_motion_radius_m']==pytest.approx(.24)
    from jev_agent_wbc.dynamic_collision import DynamicCollision
    collision=DynamicCollision(snapshot,1.4)
    assert collision.radius_rates[ids[0]]==pytest.approx(1.2+.035)
    now=collision.radii[ids[0]]
    assert now==pytest.approx(radius+.035*.4)  # growth already included in snapshot
    assert collision.at_time(.1).radii[ids[0]]==pytest.approx(now+.1235)
    from jev_agent_wbc.task_space_planning import camera_scene
    from jev_agent_wbc.camera_intent import obstacle_state
    proposal_scene=camera_scene(snapshot,1.4,3.)
    for when in (1.4,1.5,1.8):
        pc,pr=obstacle_state(proposal_scene,when)
        certificate=collision.at_time(when-1.4)
        np.testing.assert_allclose(pc,certificate.centers)
        np.testing.assert_allclose(pr,certificate.radii)
    # Every constant acceleration at or below the configured bound stays in
    # the occupied volume, even when it turns or reverses during the blind gap.
    for acceleration in ([0.,3.,0.],[-3.,0.,0.],[0.,0.,-3.]):
        actual=predicted+.5*np.array(acceleration)*.4**2
        assert np.linalg.norm(actual-predicted)+.1<=radius


@pytest.mark.parametrize('duration',[None,0.])
def test_self_mask_uses_only_postures_during_known_acquisition(duration,monkeypatch):
    mapping=DynamicLidarMap(voxel_size=.035,cluster_distance=.08)
    old=STATE.copy();current=STATE.copy();current[0]+=1.
    mapping.robot_history.append((1.,old,.002,np.eye(3)))
    seen=[]
    def distances(state,points,*args,**kwargs):
        seen.append(np.asarray(state).copy())
        return np.ones((len(points),8))
    monkeypatch.setattr(mapping.geometry,'distances',distances)
    scan=dict(points=surface([-.8,.3,.6]),rotation=np.eye(3),translation=np.zeros(3),
              state=current,base_z=.002,base_rotation=np.eye(3),stamp=1.1,sensor='front')
    if duration is not None:scan['acquisition_duration_s']=duration
    mapping.integrate_scans([scan])
    assert seen
    assert any(np.array_equal(x,current) for x in seen)
    assert any(np.array_equal(x,old) for x in seen)==(duration is None)


def observed(t, center, velocity):
    return dict(observation_stamp=t, tracks=[dict(id=1, center=center, velocity=velocity)])


def test_motion_change_is_causal_and_ignores_coherent_translation():
    a = motion_evidence(observed(0., [0., 0., .5], [.2, 0., 0.]))
    assert not motion_changed(a, motion_evidence(observed(.2, [.04, 0., .5], [.2, 0., 0.])))
    assert motion_changed(a, motion_evidence(observed(.2, [.04, 0., .5], [-.2, 0., 0.])))
    assert motion_changed(a, motion_evidence(observed(.2, [.04, .04, .5], [.2, 0., 0.])))


def test_async_preview_discards_reversal_and_immediately_requests_current_motion():
    preview = AsyncReconfigurationPreview()
    preview.reset(Reference())
    old = motion_evidence(observed(0., [1., 0., .5], [.2, 0., 0.]))
    result = ReconfigurationPreview()
    result.updated = 0.
    result.times = np.array([0., 1.]); result.controls = np.ones((2, 8))*.01
    result.states = np.tile(STATE, (2, 1)); result.info = dict(tpr_prediction_valid=True)
    preview.pending_generation = preview.generation
    preview.pending_motion = preview.requested_motion = old
    preview.requested = .1
    preview.pending = Future(); preview.pending.set_result(result)
    submissions = []
    class Pool:
        def submit(self, *args):
            submissions.append(args)
            return Future()
    preview.pool = Pool()
    c = controller()
    c.collision.scene.update(observed(.2, [1.04, 0., .5], [-.2, 0., 0.]))
    u = preview.update(c, STATE, np.zeros(8), .2, .02, 0., None, None)
    np.testing.assert_array_equal(u, np.zeros(8))
    assert preview.accepted == 0 and preview.rejected == 1
    assert preview.info['tpr_prediction_failure'] == 'observed obstacle motion changed'
    assert len(submissions) == 1  # .1 s after previous request, before periodic .4 s.
    assert preview.info['tpr_motion_replans'] == 1


def test_finite_step_recovery_keeps_actuators_and_both_safety_certificates(monkeypatch):
    # Exercise a deficient derivative reserve independently of the default
    # body-specific gains; the nonlinear safety certificates are unchanged.
    from jev_agent_wbc import body_anticipation
    monkeypatch.setattr(body_anticipation,'barrier_gains',lambda collision,indices:
        np.full((len(indices),len(collision.geometry.shape_bodies)),1.5))
    c=controller(scene((-.13,0.,.48),(.2,0.,0.)))
    c.reconfiguration.update=lambda *args:np.zeros(8)
    u,info=c.solve(STATE,*c.trajectory.sample(0.),.02)
    assert info['tpr_report']['barrier_mode']=='certified_swept_recovery'
    assert np.max(abs(u)-c.limits.acceleration*.02)<=1e-6
    assert info['command_sweep_lower_m']>=c.collision.margin
    assert info['self_clearance_lower_m']>=c.self_collision.margin


@pytest.mark.parametrize('failed_certificate',['external','self'])
def test_recovery_cannot_bypass_a_failed_nonlinear_certificate(failed_certificate,monkeypatch):
    c=controller(scene((-.13,0.,.48),(.2,0.,0.)))
    c.reconfiguration.update=lambda *args:np.zeros(8)
    if failed_certificate=='external':
        original=c.collision.sweep
        def unsafe(*args,**kwargs):
            result=original(*args,**kwargs)
            result['lower'][:]=c.collision.margin-.001
            return result
        c.collision.sweep=unsafe
        # The ramp has its own tighter certificate; both admissible proofs
        # must fail before the controller is required to reject the command.
        monkeypatch.setattr('jev_agent_wbc.tpr_wbc.certify_control_ramp',
                            lambda *args,**kwargs:c.collision.margin-.001)
    else:c.self_collision.sweep=lambda *args,**kwargs:c.self_collision.margin-.001
    with pytest.raises(RuntimeError):c.solve(STATE,*c.trajectory.sample(0.),.02)
    np.testing.assert_array_equal(c.previous,np.zeros(8))
