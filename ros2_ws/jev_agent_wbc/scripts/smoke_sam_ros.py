#!/usr/bin/env python3
"""ROS + real SAM GPU transport with a static RGB fixture, not visual control performance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('ROS_DOMAIN_ID','174')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frame',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True,help='first-frame SAM box json')
    parser.add_argument('--sam-python',default='/home/w84384651/miniconda3/envs/sam2/bin/python')
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import cv2
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image,CameraInfo
    from std_msgs.msg import String
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation
    from smoke_ros import Plant
    from jev_agent_wbc.ros_node import CinematicNode
    from jev_agent_wbc.whole_body_qp import whole_body_kinematics
    frame=cv2.cvtColor(cv2.imread(str(args.frame)),cv2.COLOR_BGR2RGB)
    height,width=frame.shape[:2]
    seed=json.loads(args.seed.read_text())['box_xyxy']
    case=args.output/'fixture';case.mkdir()
    task=json.loads((ROOT/'scenarios/movein/task.json').read_text())
    first=task['reference_camera_trajectory']['samples'][0]
    for row in task['reference_camera_trajectory']['samples']:
        row['position']=first['position'];row['quaternion_xyzw']=first['quaternion_xyzw']
    task['intent']='Hold the initial view. Static RGB transport test only.'
    fixture=json.loads((ROOT/'scenarios/movein/runtime_scene.json').read_text())
    for obstacle in fixture['obstacles']:
        for row in obstacle['keyframes']:
            row['position'][1]+=10;row['velocity']=[0,0,0]
    (case/'task.json').write_text(json.dumps(task));(case/'runtime_scene.json').write_text(json.dumps(fixture))
    cv2.imwrite(str(case/'input_frame.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    observer_log=(args.output/'observer.log').open('w')
    command=[sys.executable,'-m','jev_agent_wbc.sam_observer','--ros-args',
        '-p','subject_id:=blue_subject','-p','sam_python:='+args.sam_python,
        '-p','checkpoint:='+str(args.checkpoint.resolve()),'-p','seed_box:='+json.dumps(seed),
        '-p','observation_rate_hz:=4.0']
    observer=subprocess.Popen(command,env=env,stdout=observer_log,stderr=subprocess.STDOUT)
    rclpy.init(args=['--ros-args','-p','task_file:='+str((case/'task.json').resolve()),
        '-p','output_directory:='+str(args.output/'controller'),'-p','mode:=strict','-p','state_timeout_s:=0.5'])
    node=CinematicNode();plant=Plant(node,case)
    publisher=plant.create_publisher(Image,'/camera/image_raw',qos_profile_sensor_data)
    camera_info=plant.create_publisher(CameraInfo,'/camera/camera_info',1)
    records=[]
    plant.create_subscription(String,'/cinematic/subject_visual',lambda msg:records.append(json.loads(msg.data)),1)
    def publish_frame():
        stamp=plant.get_clock().now().to_msg()
        p,r,_=whole_body_kinematics(plant.state,plant.base_z)
        tr=TransformStamped();tr.header.stamp=stamp;tr.header.frame_id='odom';tr.child_frame_id='camera_link'
        tr.transform.translation.x,tr.transform.translation.y,tr.transform.translation.z=map(float,p)
        q=Rotation.from_matrix(r).as_quat()
        tr.transform.rotation.x,tr.transform.rotation.y,tr.transform.rotation.z,tr.transform.rotation.w=map(float,q)
        plant.tf.sendTransform(tr)
        message=Image();message.header.stamp=stamp;message.header.frame_id='camera_link'
        message.height=height;message.width=width;message.encoding='rgb8';message.step=3*width;message.data=frame.tobytes()
        info=CameraInfo();info.header=message.header;info.height=height;info.width=width
        focal=.5*height/np.tan(np.deg2rad(27.5));info.k=[float(focal),0.,width*.5,0.,float(focal),height*.5,0.,0.,1.]
        camera_info.publish(info);publisher.publish(message)
    plant.create_timer(.1,publish_frame)
    executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);executor.add_node(plant)
    started=time.monotonic();snapshots=[]
    try:
        while time.monotonic()-started<25 and not node.completed:
            executor.spin_once(timeout_sec=.02)
            if node.engine is not None:
                now=node.get_clock().now().nanoseconds*1e-9-node.start_s
                s=node.engine.perception.snapshot(now)
                if not snapshots or s['revision'] != snapshots[-1]['revision']:snapshots.append(s)
        result=dict(scope='static RGB fixture + synthetic ROS plant; real SAM inference; not a visual repair benchmark',
            started=node.engine is not None,completed=node.completed,fault=node.fault,
            sam_messages=len(records),valid_masks=sum(r.get('valid') is True for r in records),
            valid_current_visual_snapshots=sum(s['visual']['status']=='valid' for s in snapshots),
            projection_consistent_snapshots=sum(s['fusion_consistent'] is True for s in snapshots),
            scale_unknown_preserved=all(not s['scale_measurement_valid'] for s in snapshots),
            commands_received=plant.commands,wall_s=time.monotonic()-started)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'sam_evidence.json').write_text(json.dumps(records,indent=2)+'\n')
        (args.output/'fused_evidence.json').write_text(json.dumps(snapshots,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return 0 if result['completed'] and not result['fault'] and result['projection_consistent_snapshots']>0 else 2
    finally:
        executor.shutdown();node.close();node.destroy_node();plant.destroy_node();rclpy.shutdown()
        observer.terminate()
        try:observer.wait(timeout=5)
        except subprocess.TimeoutExpired:observer.kill();observer.wait(timeout=2)
        observer_log.close()


if __name__=='__main__':
    raise SystemExit(main())
