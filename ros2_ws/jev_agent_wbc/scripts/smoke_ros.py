#!/usr/bin/env python3
"""Local ROS integration smoke: synthetic runtime sensors + differential-drive plant.

This checks transport/TF/startup/control/termination, not Isaac physics or LiDAR fidelity.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('ROS_DOMAIN_ID', '173')
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation
from jev_agent_wbc.ros_node import CinematicNode
from jev_agent_wbc.scenarios import RuntimeScene
from jev_agent_wbc.sensors import RuntimeLidar, sphere_raycast, RAYS
from jev_agent_wbc.kc_svsdf import advance
from jev_agent_wbc.isaac_bridge import ARM_JOINT_NAMES, WHEEL_JOINT_NAMES


class Plant(Node):
    def __init__(self, controller, case):
        super().__init__('cinematic_test_plant')
        self.controller = controller
        self.scene = RuntimeScene(case/'runtime_scene.json')
        self.state = np.array(self.scene.document['initial_state'])
        self.base_z = self.scene.document['base_z']
        self.command = np.zeros(8)
        self.last = time.monotonic()
        self.count = self.commands = 0
        self.tf = TransformBroadcaster(self)
        self.js = self.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.odom = self.create_publisher(Odometry, '/base_controller/odom', qos_profile_sensor_data)
        self.subject = self.create_publisher(PoseStamped, '/cinematic/subject_pose', 1)
        self.clouds = [self.create_publisher(PointCloud2, topic, qos_profile_sensor_data)
                       for topic in ('/lidar/points', '/lidar/front/points')]
        self.create_subscription(JointState, '/isaac_joint_commands', self.on_command, 1)
        self.create_timer(.02, self.step)

    def on_command(self, message):
        values = dict(zip(message.name, message.velocity))
        left, right = [values[n] for n in WHEEL_JOINT_NAMES]
        self.command = np.r_[.07*(left+right)/2, .07*(right-left)/.21, [values[n] for n in ARM_JOINT_NAMES]]
        self.commands += 1

    def step(self):
        wall = time.monotonic()
        dt, self.last = min(.05, wall-self.last), wall
        self.state = advance(self.state, self.command, dt)
        stamp = self.get_clock().now().to_msg()
        now = stamp.sec+stamp.nanosec*1e-9
        t = 0. if self.controller.start_s is None else now-self.controller.start_s
        q = Rotation.from_euler('z', self.state[2]).as_quat()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = *self.state[:2], self.base_z
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q
        self.odom.publish(odom)
        joints = JointState()
        joints.header.stamp, joints.name, joints.position = stamp, list(ARM_JOINT_NAMES), self.state[3:].tolist()
        self.js.publish(joints)
        subject = PoseStamped()
        subject.header.stamp, subject.header.frame_id = stamp, 'odom'
        subject.pose.position.x, subject.pose.position.y, subject.pose.position.z = self.scene.document['subject']['position']
        subject.pose.orientation.w = 1.
        self.subject.publish(subject)
        transforms = []
        origins = RuntimeLidar.origins(self.state, self.base_z)
        for name, position, quaternion in [('base_link', [*self.state[:2], self.base_z], q),
                ('test_rear', origins[0], [0., 0., 0., 1.]), ('test_front', origins[1], [0., 0., 0., 1.])]:
            tr = TransformStamped()
            tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp, 'odom', name
            tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = [float(x) for x in position]
            tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = [float(x) for x in quaternion]
            transforms.append(tr)
        self.tf.sendTransform(transforms)
        if self.count % 5 == 0:
            wall_points = np.array([[3.5, y, z] for y in np.linspace(-1.5, 1.5, 41) for z in np.arange(.1, .8, .075)])
            for publisher, origin, frame in zip(self.clouds, origins, ('test_rear', 'test_front')):
                hits = sphere_raycast(origin, RAYS, self.scene.obstacles(t))
                points = np.asarray(np.vstack([hits, wall_points])-origin, dtype=np.float32)
                cloud = PointCloud2()
                cloud.header.stamp, cloud.header.frame_id = stamp, frame
                cloud.height, cloud.width, cloud.point_step, cloud.row_step = 1, len(points), 12, 12*len(points)
                cloud.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1) for i,n in enumerate('xyz')]
                cloud.is_dense = True
                cloud.data = points.tobytes()
                publisher.publish(cloud)
        self.count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-p', f'task_file:={ROOT}/scenarios/movein/task.json',
                    '-p', f'output_directory:={args.output}', '-p', 'mode:=strict', '-p', 'state_timeout_s:=0.5'])
    node = CinematicNode()
    plant = Plant(node, ROOT/'scenarios/movein')
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(plant)
    started = time.monotonic()
    try:
        while time.monotonic()-started < 30 and not node.completed:
            executor.spin_once(timeout_sec=.02)
        result = dict(started=node.engine is not None, completed=node.completed, fault=node.fault,
                      commands_received=plant.commands, lidar_revision=node.lidar.map.revision,
                      measurement='ROS synthetic transport integration only')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'smoke_result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        return 0 if node.completed and not node.fault and plant.commands > 30 else 2
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        plant.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
