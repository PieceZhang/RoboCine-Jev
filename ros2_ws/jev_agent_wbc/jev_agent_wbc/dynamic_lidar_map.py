"""Short-lived measured surfels, compact-object tracking, and motion prediction.

All geometry and velocities come from timestamped LiDAR returns in odom.
Scene files, object names and simulator truth are deliberately absent here.
"""
from collections import deque
from copy import copy
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .lidar_collision_map import LidarCollisionMap, voxel_keys


_worker = None


def measured_sphere(points, tolerance):
    """Accept a round-shape hypothesis only with well-conditioned surface data.

    No scene dimensions or object identities enter this fit. Non-round and
    nearly planar packets retain the generic measured-box representation.
    """
    points=np.asarray(points,float)
    if len(points)<6:
        return None
    anchor=points.mean(axis=0); local=points-anchor
    spread=np.linalg.svd(local,compute_uv=False)
    if spread[-1] < .08*spread[0]:
        return None
    fit,_,rank,_=np.linalg.lstsq(np.c_[2*local,np.ones(len(local))],np.sum(local**2,axis=1),rcond=None)
    radius2=fit[3]+fit[:3]@fit[:3]
    if rank!=4 or not .03**2<=radius2<=.45**2:
        return None
    radius=np.sqrt(radius2); center=anchor+fit[:3]
    residual=np.abs(np.linalg.norm(points-center,axis=1)-radius)
    if residual.max()>tolerance or np.linalg.norm(center-anchor)>radius:
        return None
    return dict(center=center,radius=float(radius),residual_m=float(residual.max()))


def integrate_dynamic_batch(settings, scans):
    global _worker
    if _worker is None:
        _worker = DynamicLidarMap(**settings)
    started = time.monotonic()
    latest = {scan['sensor']: scan for scan in scans}
    _worker.integrate_scans(list(latest.values()))
    # The ROS process consumes a scene snapshot, never the worker's scan queues
    # or static-evidence database. Transferring those growing histories on every
    # batch stalls the control executor while Python unpickles them under GIL.
    snapshot = copy(_worker)
    snapshot.latest = {}
    snapshot.robot_history = deque()
    snapshot.world_support = {}
    snapshot.static_world_tree = None
    snapshot.observations = {}
    snapshot.tracks = {key: {name: value for name, value in track.items()
                            if name not in ('history', 'template')}
                       for key, track in _worker.tracks.items()}
    return dict(map=snapshot, update_ms=1000 * (time.monotonic() - started))


class DynamicLidarMap(LidarCollisionMap):
    def __init__(self, *, fusion_window_s=.30, track_timeout_s=2.5,
                 cluster_distance=.16, velocity_window_s=1.8,
                 maximum_speed=1., tracking_radius=2.5, **kwargs):
        super().__init__(**kwargs, confirmations=1)
        self.fusion_window = float(fusion_window_s)
        self.track_timeout = float(track_timeout_s)
        self.cluster_distance = float(cluster_distance)
        self.velocity_window = float(velocity_window_s)
        self.maximum_speed = float(maximum_speed)
        self.tracking_radius = float(tracking_radius)
        if not np.isfinite([self.fusion_window, self.track_timeout, self.cluster_distance,
                           self.velocity_window, self.maximum_speed, self.tracking_radius]).all() or min(
                self.fusion_window, self.track_timeout, self.cluster_distance,
                self.velocity_window, self.maximum_speed, self.tracking_radius) <= 0:
            raise ValueError('invalid dynamic mapping parameters')
        self.latest = {}
        self.robot_history = deque()
        self.robot_xy = np.zeros(2)
        self.tracks = {}
        self.next_id = 1
        self.stamp = 0.
        self.last_track_stamp = -np.inf
        self.moving_observations = 0
        self.expired_tracks = 0
        self.world_support = {}
        self.static_world_tree = None

    def integrate_scans(self, scans):
        if not scans:
            return
        stamp = max(scan['stamp'] for scan in scans)
        if stamp <= self.stamp:
            return
        for scan in scans:
            if not self.robot_history or scan['stamp'] - self.robot_history[-1][0] >= .15:
                self.robot_history.append((scan['stamp'], scan['state'].copy(),
                                           scan['base_z'], scan['base_rotation'].copy()))
            while self.robot_history and stamp - self.robot_history[0][0] > 1.8:
                self.robot_history.popleft()
            self.robot_xy = scan['state'][:2].copy()
            # Reuse the established timestamped articulated self/range/height
            # filter, but never accumulate a dynamic object's historical trail.
            filtered = LidarCollisionMap(voxel_size=self.voxel_size, map_radius=self.map_radius,
                minimum_height=self.minimum_height, maximum_height=self.maximum_height,
                minimum_clearance=self.minimum_clearance, retention_s=self.retention_s, confirmations=1)
            filtered.integrate(**scan)
            points = filtered.filtered_points.copy()
            # A rotating packet may contain the robot at an earlier posture.
            # Its own articulated surface must not become a retained obstacle
            # when the joints move. Keep this filter local and use measured TF
            # and joints only; no simulator body labels or truth are available.
            near = np.flatnonzero(np.linalg.norm(points[:, :2] - self.robot_xy, axis=1) < 2.)
            keep = np.ones(len(points), dtype=bool)
            for _, state, z, rotation in self.robot_history:
                if len(near):
                    own_distance = self.geometry.distances(state, points[near], z, rotation).min(axis=1)
                    keep[near] &= own_distance > .10
            # A return beyond an opaque robot link cannot be an external
            # surface if its measured ray traverses that link. Check only
            # near-body candidates; distant wall rays remain inexpensive.
            if len(near):
                distance = self.geometry.distances(scan['state'], points[near],
                    scan['base_z'], scan['base_rotation']).min(axis=1)
                shadow = near[keep[near] & (distance < .30)]
                if len(shadow):
                    fractions = np.linspace(.12, .98, 41)
                    origin = np.asarray(scan['translation'])
                    rays = origin + (points[shadow, None] - origin) * fractions[None, :, None]
                    blocked = self.geometry.distances(scan['state'], rays.reshape(-1, 3),
                        scan['base_z'], scan['base_rotation']).min(axis=1).reshape(len(shadow), -1)
                    keep[shadow] &= blocked.min(axis=1) >= -.015
            points = points[keep]
            if filtered.bad_scan is not None:
                self.bad_scan = filtered.bad_scan
            recent = self.latest.setdefault(scan['sensor'], deque())
            recent.append((scan['stamp'], points))
            while recent and stamp - recent[0][0] > self.fusion_window:
                recent.popleft()
            self.scans[scan['sensor']] = self.scans.get(scan['sensor'], 0) + 1
        chunks = [(t, points) for recent in self.latest.values() for t, points in recent
                  if stamp - t <= self.fusion_window and len(points)]
        if not chunks:
            return
        current_chunks=[points for measured_at,points in chunks if abs(measured_at-stamp)<1e-6]
        current=np.vstack(current_chunks) if current_chunks else np.empty((0,3))
        self.current_points=current
        points = self.align_measurements(chunks, stamp)
        _, indices = voxel_keys(points, self.voxel_size)
        points = points[indices]
        # Single noisy returns, even when repeated, are not supported surface
        # patches. Require three neighbouring measured voxels before fusion.
        supported = cKDTree(points).query_ball_point(points, max(.10, 2.5 * self.voxel_size),
                                                     return_length=True) >= 3
        points = points[supported]
        self.stamp = stamp
        if stamp - self.last_track_stamp >= .09:
            # Confirm surfaces at fixed odom coordinates across time. A new
            # window on a sofa can have a moving centroid while its actual
            # surface points still coincide with this measured static history.
            dynamic = [track for track in self.tracks.values() if track['moving']]
            stationary = np.ones(len(points), bool)
            for track in dynamic:
                predicted = track['center'] + track['velocity']*(stamp-track['stamp'])
                stationary &= np.any(np.abs(points-predicted) > track['model_extent']+.08, axis=1)
            for key in map(tuple, np.floor(points[stationary]/self.voxel_size).astype(int)):
                first, _, count = self.world_support.get(key, (stamp, stamp, 0))
                self.world_support[key] = (first, stamp, count+1)
            self.world_support = {key: value for key, value in self.world_support.items() if stamp-value[1] < 30.}
            fixed = [(np.array(key)+.5)*self.voxel_size for key, (first, last, count) in self.world_support.items()
                     if last-first >= 1.5 and count >= 4]
            self.static_world_tree = cKDTree(fixed) if fixed else None
            self.track(points, stamp)
            self.last_track_stamp = stamp
        # A bounded local map is reconstructed from current observations.
        # Only static cells have retention; tracked movers never enter it.
        dynamic = [track for track in self.tracks.values() if track['moving']]
        static_points = points.copy()
        for track in dynamic:
            extent = track['extent'] + .12
            static_points = static_points[np.any(np.abs(static_points - track['center']) > extent, axis=1)]
        for key in map(tuple, np.floor(static_points / self.voxel_size).astype(int)):
            count, previous = self.observations.get(key, (0, -np.inf))
            if stamp - previous >= .03:
                count = min(2, count + 1) if stamp - previous < 1. else 1
                self.observations[key] = (count, stamp)
            if count >= 2 or key in self.cells:
                self.cells[key] = stamp
        self.observations = {key: observation for key, observation in self.observations.items()
                             if stamp - observation[1] < 1.}
        if self.cells:
            keys = np.array(list(self.cells))
            seen = np.fromiter(self.cells.values(), float)
            centers = (keys + .5) * self.voxel_size
            keep = stamp - seen <= self.retention_s
            keep &= np.linalg.norm(centers[:, :2] - scans[-1]['state'][:2], axis=1) <= self.map_radius
            for track in dynamic:
                # Remove both the current envelope and its observed history.
                # This also clears cells seen before the mover was classified.
                for _, center in track['history']:
                    keep &= np.any(np.abs(centers - center) > track['extent'] + .12, axis=1)
            self.cells = dict(zip(map(tuple, keys[keep]), seen[keep]))
        self.revision += 1

    def align_measurements(self, chunks, stamp):
        result = []
        for measured_at, points in chunks:
            aligned = points.copy()
            claimed = np.zeros(len(points), bool)
            for track in self.tracks.values():
                if not track['moving']:
                    continue
                center = track['center']+track['velocity']*(measured_at-track['stamp'])
                selected = ~claimed & np.all(np.abs(points-center) <= track['model_extent']+.06, axis=1)
                aligned[selected] += track['velocity']*(stamp-measured_at)
                claimed[selected] = True
            result.append(aligned)
        return np.vstack(result)

    def clusters(self, points):
        if not len(points):
            return []
        pairs = cKDTree(points).query_pairs(self.cluster_distance, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
        count, labels = connected_components(graph, directed=False)
        groups = []
        for label in range(count):
            selected = points[labels == label]
            if len(selected) < 4:
                continue
            low, high = selected.min(axis=0), selected.max(axis=0)
            if np.linalg.norm((low[:2] + high[:2]) / 2 - self.robot_xy) > self.tracking_radius:
                continue
            # Walls/furniture networks remain static surfels. Compact objects
            # are associated in odom, so robot ego-motion is already removed.
            current=getattr(self,'current_points',selected)
            packet=current[np.all((current>=low-.08)&(current<=high+.08),axis=1)]
            round_fit=measured_sphere(packet,min(.005,.2*self.voxel_size))
            if np.max(high[:2] - low[:2]) > .85 or (high[2] - low[2] < .10 and round_fit is None):
                continue
            span = high - low
            # Rotating partial scans translate the visible packet along a
            # wall. A tall, wide, thin sheet has no bounded object evidence;
            # keep it as static collision surfels instead of a moving track.
            if (span[2] > .60 and np.min(span[:2]) < max(.05, 1.5 * self.voxel_size)):
                continue
            groups.append(dict(center=(low + high) / 2, extent=(high - low) / 2, points=selected))
        return groups

    def track(self, points, stamp):
        expired = [key for key, track in self.tracks.items() if stamp - track['stamp'] > self.track_timeout]
        for key in expired:
            del self.tracks[key]
            self.expired_tracks += 1
        # Segment an established mover inside its predicted, measured object
        # envelope before clustering the remaining cloud. Otherwise a post
        # passing close to a table merges into its large connected component,
        # loses its identity, and becomes an erroneously stationary obstacle.
        detections, assigned = [], {}
        remaining = np.ones(len(points), bool)
        moving_ids = {key for key, track in self.tracks.items() if track['moving']}
        for key in sorted(moving_ids):
            track = self.tracks[key]
            predicted = track['center'] + track['velocity']*(stamp-track['stamp'])
            selected = remaining & np.all(np.abs(points-predicted) <= track['model_extent']+.05, axis=1)
            if np.sum(selected) < 4:
                continue
            measured = points[selected]
            low, high = measured.min(axis=0), measured.max(axis=0)
            # A partial visible packet cannot recenter a rigid object's box.
            # Intersect the measured extrema with the retained half extent;
            # keep the causal predicted center whenever it is consistent.
            # Recentring on a clipped packet manufactures lateral velocity as
            # the robot's self mask hides one side of an approaching mover.
            center = np.clip(predicted, high-track['model_extent'], low+track['model_extent'])
            center[2] = track['model_center_z']
            assigned[len(detections)] = key
            detections.append(dict(center=center, extent=track['model_extent'], points=measured))
            remaining[selected] = False
        detections.extend(self.clusters(points[remaining]))
        # Fit only simultaneous raw returns: accumulated asynchronous packets
        # stretch a moving sphere and corrupt both its center and velocity.
        for detection in detections:
            current=getattr(self,'current_points',points)
            packet=current[np.all(abs(current-detection['center']) <= detection['extent']+.08,axis=1)]
            fitted=measured_sphere(packet,min(.005,.2*self.voxel_size))
            if fitted is not None:
                detection.update(center=fitted['center'],extent=np.full(3,fitted['radius']),sphere=fitted)
        identifiers = [key for key in self.tracks if key not in moving_ids]
        if identifiers and detections:
            available = [index for index in range(len(detections)) if index not in assigned]
            costs = np.array([[np.linalg.norm(detections[index]['center'] - (self.tracks[key]['center'] +
                self.tracks[key]['velocity'] * (stamp - self.tracks[key]['stamp']))) +
                .5 * np.linalg.norm(detections[index]['extent'] - self.tracks[key]['extent']) for index in available]
                for key in identifiers])
            rows, cols = linear_sum_assignment(costs) if available else ([], [])
            assigned.update({available[int(col)]: identifiers[row] for row, col in zip(rows, cols)
                        if costs[row, col] < .22 + min(.45,
                            .25 * (stamp - self.tracks[identifiers[row]]['stamp']))})
        for index, detection in enumerate(detections):
            key = assigned.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = dict(history=deque(), velocity=np.zeros(3), moving=False,
                                        anchor=detection['center'].copy(), first_stamp=stamp,
                                        template=detection['points'].copy(), motion_votes=0)
            track = self.tracks[key]
            if track.get('sphere') is not None and 'sphere' not in detection:
                radius=track['sphere']['radius']
                if np.any(np.linalg.norm(detection['points']-detection['center'],axis=1)>radius+2*self.voxel_size):
                    track.pop('sphere')
            # A bounding-box centre can move when a static object becomes
            # partially occluded. Require novel surfaces relative to its
            # accumulated stationary template as well as coherent translation.
            novelty = float(np.mean(cKDTree(track['template']).query(detection['points'])[0]
                                    > self.voxel_size * 1.25))
            static_fraction = (float(np.mean(self.static_world_tree.query(detection['points'])[0] <= self.voxel_size*1.25))
                               if self.static_world_tree is not None else 0.)
            track['static_fraction'] = static_fraction
            track['history'].append((stamp, detection['center'].copy()))
            while track['history'] and stamp - track['history'][0][0] > self.velocity_window:
                track['history'].popleft()
            if len(track['history']) >= 3:
                times, centers = zip(*track['history'])
                times = np.array(times) - np.mean(times)
                centers = np.array(centers)
                velocity = times @ centers / max(times @ times, 1e-9)
                # A well-conditioned rigid sphere fit separates actual 3-D
                # translation from changing surface coverage. Generic partial
                # boxes still lack that vertical-motion evidence.
                sphere_observed='sphere' in detection
                if not sphere_observed:velocity[2] = 0.
                local_velocity=False
                track['acceleration']=None
                if sphere_observed and len(times)>=4:
                    # A round object can accelerate at a crossing. A long-window
                    # secant plus a 0.3 m/s² limiter retained an obsolete velocity
                    # after measured braking. Fit the latest simultaneous centers;
                    # accept only a well-conditioned, low-residual local model.
                    recent=times>=times[-1]-.4-1e-8
                    local_t=times[recent]-times[-1]
                    design=np.column_stack([np.ones(len(local_t)),local_t,.5*local_t**2])
                    coefficients,_,rank,_=np.linalg.lstsq(design,centers[recent],rcond=None)
                    residual_m=np.linalg.norm(design@coefficients-centers[recent],axis=1).max()
                    if rank==3 and len(local_t)>=4 and residual_m<.005 and np.linalg.norm(coefficients[2])<=3.:
                        velocity=coefficients[1];local_velocity=True
                        track['acceleration']=coefficients[2].copy()
                speed = np.linalg.norm(velocity)
                if speed > self.maximum_speed:
                    velocity *= self.maximum_speed / speed
                if track['moving'] and not local_velocity:
                    interval = max(.01, stamp - track['stamp'])
                    change = velocity - track['velocity']
                    length = np.linalg.norm(change)
                    if length > .3 * interval:
                        velocity = track['velocity'] + change * (.3 * interval / length)
                track['velocity'] = velocity
                dimensions=3 if sphere_observed else 2
                displacement = np.linalg.norm(detection['center'][:dimensions] - track['anchor'][:dimensions])
                # Motion classification tests translation coherence, not whether
                # its current derivative equals the full-window average speed.
                classification_times=times
                classification_centers=centers
                if sphere_observed:
                    recent=times>=times[-1]-.3-1e-8
                    classification_times=times[recent]-times[recent].mean()
                    classification_centers=centers[recent]
                average_velocity=classification_times@classification_centers/max(classification_times@classification_times,1e-9)
                residual = classification_centers[:, :dimensions] - classification_centers[:, :dimensions].mean(axis=0) - classification_times[:, None] * average_velocity[:dimensions]
                energy = np.sum((classification_centers[:, :dimensions] - classification_centers[:, :dimensions].mean(axis=0)) ** 2)
                consistency = 1. - np.sum(residual ** 2) / max(energy, 1e-9)
                evidence = (len(detection['points']) >= (6 if sphere_observed else 12) and displacement > (.008 if sphere_observed else .09) and speed > .055 and np.ptp(classification_times) > (.09 if sphere_observed else .3)
                            and consistency > .65 and (novelty > .25 or sphere_observed) and static_fraction < .25)
                track['motion_votes'] = track['motion_votes'] + 1 if evidence else 0
                if not track['moving'] and track['motion_votes'] >= 2:
                    track['moving'] = True
                    track['model_extent'] = detection['extent'].copy()
                    track['model_center_z'] = detection['center'][2]
            if (not track['moving'] and novelty < .25 and
                    (stamp - track['first_stamp'] < .6 or
                     (np.linalg.norm(detection['center'] - track['anchor']) < .04 and
                      np.linalg.norm(track['velocity']) < .035))):
                template = np.vstack([track['template'], detection['points']])
                _, keep = voxel_keys(template, self.voxel_size)
                track['template'] = template[keep]
            track.update(detection, stamp=stamp)
            if track['moving']:
                self.moving_observations += 1

    def track_diagnostics(self):
        # The recording stamp is the fused map time, including occluded tracks.
        # Keep the measured pose/time as well as the current predicted pose so
        # offline truth matching never compares an old detection to a new time.
        records = []
        for key, track in self.tracks.items():
            age = max(0., self.stamp-track['stamp'])
            records.append(dict(id=key,
                center=(track['center']+track['velocity']*age).tolist(),
                measured_center=track['center'].tolist(), measurement_stamp=float(track['stamp']),
                prediction_age_s=age, velocity=track['velocity'].tolist(),
                moving=bool(track['moving']), votes=track['motion_votes'], surfaces=len(track['points'])))
        return records

    def snapshot(self):
        static = super().snapshot()
        centers = [static['surfel_centers']]
        velocities = [np.zeros_like(centers[0])]
        ages = [np.zeros(len(centers[0]))]
        ids = [np.zeros(len(centers[0]), dtype=int)]
        radii = [np.full(len(centers[0]), static['surfel_radius'])]
        tracks = []
        objects=[]
        for key,track in self.tracks.items():
            age=max(0.,self.stamp-track['stamp'])
            if age>self.track_timeout:continue
            moving=bool(track['moving'])
            center=track['center']+(track['velocity']*age if moving else 0.)
            extent=track['model_extent'] if 'model_extent' in track else track['extent']
            objects.append(dict(id=key,center=center.tolist(),half_extent=extent.tolist(),
                height_range_m=[float(center[2]-extent[2]),float(center[2]+extent[2])],
                moving_confirmed=moving,velocity=(track['velocity'] if moving else np.zeros(3)).tolist(),
                velocity_status='measured_motion' if moving else 'motion_not_yet_confirmed',age_s=age,
                shape_model='measured_sphere' if track.get('sphere') is not None else 'measured_box',
                source='timestamped_lidar_returns_only'))
            if (not moving and track.get('sphere') is not None and track.get('motion_votes',0)>=1
                    and np.linalg.norm(track['velocity'])>.04):
                objects[-1]['motion_hint']=dict(velocity_estimate=track['velocity'].tolist(),
                    confirmation_votes=track['motion_votes'],confirmed=False,
                    source='coherent_measured_sphere_centers_pending_second_confirmation')
        for key, track in self.tracks.items():
            if not track['moving'] or self.stamp - track['stamp'] > self.track_timeout:
                continue
            age = max(0., self.stamp - track['stamp'])
            # Retain the most recent measured surface during a short occlusion.
            # Recentring and accumulating every historical bounding box would
            # expand a rigid object when its visible side changes.
            points = track['points'] + track['velocity'] * age
            # A mover must not become hollow when close/occluded returns are
            # rejected by the robot self mask. Preserve its measured compact
            # extent with a conservative enclosing volume, independent of the
            # current visible surfels. No simulator dimensions enter this bound.
            center = track['center']+track['velocity']*age
            # Eight balls cover the eight sub-boxes of the retained extent.
            # One circumsphere creates excessive empty occupied corners around
            # long/thin objects; partitioning keeps the same containment claim.
            corners = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
            extent = np.maximum(track['model_extent'], np.max(abs(track['points']-track['center']), axis=0))
            half_extent = .5*extent
            proxies = center+corners*half_extent
            proxy_radius = np.linalg.norm(half_extent)+static['surfel_radius']
            if track.get('sphere') is not None:
                # Retain an observed round rigid shape through partial
                # occlusion, with the same range/voxel uncertainty allowance.
                proxies=center[None]
                proxy_radius=track['sphere']['radius']+static['surfel_radius']+track['sphere']['residual_m']
            # The proxy union contains the retained AABB AND all current
            # measured surfel balls. Use just these eight witnesses; retaining
            # hundreds of redundant interior surfels slows the safety QP.
            count = len(proxies)
            centers.append(proxies)
            velocities.append(np.tile(track['velocity'], (count, 1)))
            ids.append(np.full(count, key))
            ages.append(np.full(count, age))
            radii.append(np.full(len(proxies), proxy_radius))
            tracks.append(dict(id=key, center=(track['center'] + track['velocity'] * age).tolist(),
                velocity=track['velocity'].tolist(), age_s=age, surfaces=len(points),
                acceleration=(track['acceleration'].tolist() if track.get('acceleration') is not None else None),
                acceleration_source='recent_observed_sphere_centers_quadratic_fit' if track.get('acceleration') is not None else 'unknown',
                shape_model='measured_sphere' if track.get('sphere') is not None else 'measured_box'))
        return {
            **static, 'surfel_centers': np.vstack(centers), 'surfel_velocities': np.vstack(velocities),
            'surfel_track_ids': np.concatenate(ids), 'surfel_ages': np.concatenate(ages), 'surfel_radii': np.concatenate(radii),
            'observation_stamp': self.stamp, 'tracks': tracks,'object_observations':objects}
