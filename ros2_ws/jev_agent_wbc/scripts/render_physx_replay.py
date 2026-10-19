#!/usr/bin/env python3
"""Render a completed no-RGB run's measured link poses without running physics.

This creates presentation artifacts only. It never reruns the Agent, overwrites
physical evidence, or treats the rendering time as real-time execution evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fps',type=float,default=8.)
    p.add_argument('--width',type=int,default=384)
    p.add_argument('--height',type=int,default=288)
    args=p.parse_args()
    folder=args.run.resolve()
    if (folder/'video').exists():p.error('video already exists; preserve existing evidence')
    from isaacsim import SimulationApp
    sys.argv=[sys.argv[0]]
    app=SimulationApp(dict(headless=True,width=args.width,height=args.height,
                           renderer='RaytracedLighting',anti_aliasing=0,multi_gpu=False,limit_cpu_threads=8))
    recorder=None
    try:
        import numpy as np
        import omni.usd
        import carb
        import omni.physx
        from pxr import Gf,UsdGeom,UsdPhysics
        from scipy.spatial.transform import Rotation
        from types import SimpleNamespace
        from run_isaac import RGBPair
        from jev_agent_wbc.reference import OriginalReference
        replay=folder/'replay'
        manifest=json.loads((replay/'manifest.json').read_text())
        if manifest['protocol']!='measured_link_poses/v1' or not manifest['measured']:
            raise ValueError('requires measured physical link poses')
        data=np.load(replay/'measured_links.npz',allow_pickle=False)
        paths=data['paths'].tolist();poses=data['transforms'];times=data['times']
        rows=json.loads((folder/'telemetry.json').read_text())
        np.testing.assert_array_equal(times,np.array([r['t'] for r in rows]))
        if poses.shape != (len(rows),len(paths),7):raise ValueError('link pose shape mismatch')
        original=json.loads((folder/'runtime_scene.json').read_text())
        reference=OriginalReference(folder/'original_task.json',original['subject']['position'])
        context=omni.usd.get_context()
        context.open_stage(str(replay/'scene.usdc'))
        stage=context.get_stage()
        physical_steps=[]
        step_subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(
            lambda dt:physical_steps.append(float(dt)))
        # No integrator, joints or drives are used in this presentation stage.
        # These attributes prevent a rendering update from advancing actors.
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        for _ in range(3):app.update()
        operations=[]
        for path in paths:
            prim=stage.GetPrimAtPath(path)
            if not prim:raise ValueError(f'missing measured link {path}')
            xf=UsdGeom.Xformable(prim)
            operation=xf.MakeMatrixXform();xf.SetResetXformStack(True)
            operations.append(operation)
        movers={name:UsdGeom.Xformable(stage.GetPrimAtPath('/World/MovingDisplay/'+name)).MakeMatrixXform()
                for name in rows[0]['measured_obstacles']}
        class RenderClock:
            current_time=0.
            def render(self):
                carb.settings.get_settings().set('/app/player/playSimulations',False)
                app.update()
        clock=RenderClock()
        options=SimpleNamespace(record=True,record_fps=args.fps,width=args.width,height=args.height,
            render_subframes=1,camera_output='execution',publish_camera_ros=False,
            skip_reference_video=False)
        recorder=RGBPair(stage,clock,options,folder,str(Path(paths[0]).parent),original['subject'])
        # A rendered replay is not a new optical/raycast observation.
        recorder.visibility=lambda *args:dict(available=False,reason='offline_state_replay_no_new_measurement')
        max_position=max_rotation=0.
        next_frame=0.
        for index,row in enumerate(rows):
            if row['t']+1e-9<next_frame:continue
            for op,pose in zip(operations,poses[index]):
                quat=Rotation.from_quat(pose[3:]).as_quat()
                matrix=Gf.Matrix4d(1.)
                matrix.SetRotate(Gf.Quatd(float(quat[3]),Gf.Vec3d(*map(float,quat[:3]))))
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
                op.Set(matrix)
            for name,op in movers.items():
                op.Set(Gf.Matrix4d(1.).SetTranslate(Gf.Vec3d(*row['measured_obstacles'][name])))
            clock.current_time=float(row['t'])
            # Verify the actual authored world transforms, not just setter inputs.
            for path,pose in zip(paths,poses[index]):
                world=UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
                position=np.asarray(world.ExtractTranslation())
                rotation=world.ExtractRotationQuat()
                quat=np.r_[np.asarray(rotation.GetImaginary()),float(rotation.GetReal())]
                max_position=max(max_position,float(np.linalg.norm(position-pose[:3])))
                max_rotation=max(max_rotation,float((Rotation.from_quat(quat)*Rotation.from_quat(pose[3:]).inv()).magnitude()))
            rp,rr,_=reference.sample(row['t'])
            recorder.capture(row['t'],rp,rr,np.asarray(row['actual_camera_position']),
                Rotation.from_quat(row['actual_camera_quaternion_xyzw']).as_matrix(),row['subject_track'])
            frame=recorder.frame_metadata[-1]
            frame.update(capture_protocol='measured_state_replay_same_timestamp',
                physical_time_before_capture=None,physical_time_after_capture=None,
                replay_time_before_capture=row['t'],replay_time_after_capture=row['t'],
                physical_source_sample=index,live_sensor_capture=False)
            next_frame+=1/args.fps
        recorder.close();recorder=None
        video=folder/'video'
        frames=json.loads((video/'frames.json').read_text())
        (video/'frames.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in frames))
        metadata=json.loads((video/'reference_rendering.json').read_text())
        metadata.update(protocol='measured_state_replay_original_reference_same_timestamp/v1',
            live_sensor_capture=False,measured_state_replay=True,
            scope='offline redraw of measured PhysX link poses at their original timestamps; no new Agent or physics execution',
            max_link_position_error_m=max_position,max_link_rotation_error_rad=max_rotation,
            valid=metadata['valid'] and max_position<1e-7 and max_rotation<1e-7 and not physical_steps)
        (video/'reference_rendering.json').write_text(json.dumps(metadata,indent=2)+'\n')
        camera=json.loads((video/'camera_output.json').read_text())
        camera.update(capture_mode='offline_measured_state_replay',
            raw_role='replay of the raw camera pose offered to the Agent; no new Agent inference')
        (video/'camera_output.json').write_text(json.dumps(camera,indent=2)+'\n')
        audit=dict(**metadata,physical_run_summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
            measured_links_sha256=hashlib.sha256((replay/'measured_links.npz').read_bytes()).hexdigest(),
            scene_sha256=hashlib.sha256((replay/'scene.usdc').read_bytes()).hexdigest(),
            frame_count=len(frames),physics_steps_during_render=len(physical_steps),
            source='measured PhysX state replay; distinct from live camera recording')
        (video/'replay_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        del step_subscription
        print(json.dumps(audit),flush=True)
    finally:
        if recorder is not None:recorder.close()
        app.close()


if __name__=='__main__':main()
