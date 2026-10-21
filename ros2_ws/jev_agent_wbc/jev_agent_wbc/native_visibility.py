"""Equivalent native arithmetic for the shared measured visibility representation.

The 13 x 17 silhouette rays, cylinder surface and strict sphere occlusion test
are unchanged. IntentSpec still owns every cinematic score and threshold.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .camera_intent import cylinder_points,project,obstacle_state,nominal_visibility_scene,POLICY,progress_at
from .native_geometry_kernel import AVAILABLE


def _rays(position,rotation,target_rotation,subject,radius,height,intrinsics,low,high,centers,radii):
    origin=target_rotation.T@(position-subject)
    fx,fy,cx,cy=intrinsics
    count=0;visible=0;contained=0
    for iy in range(17):
        v=low[1]+(iy+.5)/17*(high[1]-low[1])
        for ix in range(13):
            u=low[0]+(ix+.5)/13*(high[0]-low[0])
            ray=rotation@np.array([1.,(cx-u)/fx,(cy-v)/fy])
            local=target_rotation.T@ray
            a=local[0]*local[0]+local[1]*local[1]
            b=2*(local[0]*origin[0]+local[1]*origin[1])
            c=origin[0]*origin[0]+origin[1]*origin[1]-radius*radius
            discriminant=b*b-4*a*c;hit=np.inf
            for sign in (-1.,1.):
                root=(-b+sign*np.sqrt(max(0.,discriminant)))/max(2*a,1e-12)
                z=origin[2]+root*local[2]
                if discriminant>=0. and root>0. and abs(z)<=height/2:hit=min(hit,root)
            for z in (-height/2,height/2):
                root=(z-origin[2])/(local[2] if abs(local[2])>1e-12 else 1e-12)
                x=origin[0]+root*local[0];y=origin[1]+root*local[1]
                if root>0. and x*x+y*y<=radius*radius:hit=min(hit,root)
            if not np.isfinite(hit):continue
            count+=1
            if 0.<=u<=1. and 0.<=v<=1.:contained+=1
            delta=ray*hit;target=position+delta
            blocked=False
            for k in range(len(centers)):
                center=centers[k];r=radii[k]
                # Exact per-segment AABB rejection, a subset of the shared
                # all-ray AABB rejection. No potential intersection is lost.
                possible=True
                for axis in range(3):
                    if center[axis]+r<min(position[axis],target[axis]) or center[axis]-r>max(position[axis],target[axis]):
                        possible=False;break
                if not possible:continue
                fraction=min(1.,max(0.,np.dot(delta,center-position)/max(np.dot(delta,delta),1e-12)))
                difference=position+fraction*delta-center
                if np.sqrt(np.dot(difference,difference))<r:
                    blocked=True;break
            if not blocked:visible+=1
    return count,visible,contained


if AVAILABLE:
    from numba import njit
    _rays=njit(cache=True,nogil=True)(_rays)


def prewarm():
    _rays(np.zeros(3),np.eye(3),np.eye(3),np.array([2.,0.,0.]),.2,.5,
          np.array([.5,.5,.5,.5]),np.array([.4,.4]),np.array([.6,.6]),
          np.array([[1.,.5,0.]]),np.array([.1]))


def visibility(position,rotation,scene,t,model,intrinsics):
    if model is None or not scene.get('valid'):
        return dict(visibility=None,in_frame=None,projected_area=None,source='unknown_subject_geometry_or_obstacles')
    uv,depth=project(cylinder_points(model),position,rotation,intrinsics)
    if np.any(depth<=1e-5):
        return dict(visibility=0.,in_frame=0.,projected_area=None,source='behind_camera')
    area=float(ConvexHull(uv).volume)
    target_rotation=Rotation.from_quat(model.get('orientation_xyzw',[0.,0.,0.,1.])).as_matrix()
    centers,radii=obstacle_state(scene,t)
    count,visible,contained=_rays(np.asarray(position,float),np.asarray(rotation,float),target_rotation,
        np.asarray(model['position'],float),float(model['radius']),float(model['height']),
        np.asarray(intrinsics,float),uv.min(axis=0),uv.max(axis=0),centers,radii)
    if not count:
        return dict(visibility=None,in_frame=None,projected_area=area,source='no_surface_rays')
    return dict(visibility=visible/count,in_frame=contained/count,projected_area=area,surface_rays=count,
                source='calibrated_cylinder_silhouette_grid_sphere_occlusion')


def intent_prediction(spec,path,scene,t,history=(),step=.05):
    """The existing prediction grid/scoring with equivalent visibility arithmetic."""
    from .intent_paths import poses_at
    from .floating_camera import camera_motion_limits
    end=path.duration+POLICY.terminal_hold_s
    boundaries=[end]+([path.duration] if path.duration>=t else [])
    times=np.unique(np.round(np.r_[np.arange(t,end-1e-8,step),boundaries],9))
    pp,rr=poses_at(path,times);poses=list(zip(pp,rr))
    previous=[row for row in history if row['t']<t-1e-8]
    positions=[row['camera_position'] for row in previous]+[p[0] for p in poses]
    rotations=[Rotation.from_quat(row['camera_quaternion_xyzw']).as_matrix() for row in previous]+[p[1] for p in poses]
    progress=[row['progress_s'] for row in previous]+[progress_at(path,at) for at in times]
    view=nominal_visibility_scene(scene)
    measurements=[row['visual'] for row in previous]+[
        visibility(p[0],p[1],view,float(at),spec.model,spec.intrinsics) for at,p in zip(times,poses)]
    result=spec.score(np.r_[[row['t'] for row in previous],times],positions,rotations,measurements,progress)
    result['visibility_forecast']=dict(mode='nominal_observed_geometry_and_velocity',guarantee=False,
        physical_uncertainty_envelope_retained=True,max_future_radius_growth_m=float(
            max(0.,end-scene['observation_stamp'])*np.max(scene.get('radius_rates',[0.]),initial=0.)))
    result.update(camera_motion_limits(poses,times))
    return result
