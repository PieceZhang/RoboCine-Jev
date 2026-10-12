"""Physics feedback preserves precision, one timestamp and the odometry frame."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('rclpy')
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from jvs_mock.joint_command_filter import JointCommandFilter

SCRIPT=Path(__file__).resolve().parents[1]/'ros2_ws/jvs_mock/scripts/jvs_isaac_sim.py'
spec=importlib.util.spec_from_file_location('physics_feedback_test_runner',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Publisher:
    def __init__(self,label,events):self.label,self.events=label,events
    def publish(self,message):self.events.append((self.label,message))


class Robot:
    def __init__(self,position,rotation):self.position,self.rotation=position,rotation
    def get_world_pose(self):
        x,y,z,w=self.rotation.as_quat();return self.position,np.array([w,x,y,z])
    def get_joint_positions(self):return np.array([.123456789,-.765432109])
    def get_joint_velocities(self):return np.array([.234567891,-.345678912])
    def get_linear_velocity(self):return self.rotation.apply([.4,0.,0.])
    def get_angular_velocity(self):return self.rotation.apply([0.,0.,.8])


def feedback(offset,yaw):
    item=runner.PhysicsFeedback.__new__(runner.PhysicsFeedback)
    item.np,item.rotation=np,Rotation
    item.joint_type,item.odom_type,item.clock_type=JointState,Odometry,Clock
    item.transform_type,item.tf_type=TransformStamped,TFMessage
    item.origin=np.array([5.8,-.5,.002]);initial=Rotation.from_euler('z',-1.92)
    item.initial_inverse=initial.inv()
    item.robot=Robot(item.origin+initial.apply(offset),initial*Rotation.from_euler('z',yaw))
    item.names=['joint1','joint2'];item.time=item.next_time=0.;item.period=.02
    item.samples,item.error=0,'';events=[]
    for attribute,label in [('joints','joints'),('odom','odom'),('tf','tf'),('clock','clock')]:
        setattr(item,attribute,Publisher(label,events))
    return item,events


def test_same_physical_sample_has_one_timestamp_and_no_four_decimal_rounding():
    item,events=feedback([0.,0.,0.],0.)
    item.on_step(.001)
    assert item.error=='' and item.samples==1
    assert [label for label,_ in events]==['joints','odom','tf','clock']
    data=dict(events);stamp=data['clock'].clock
    assert data['joints'].header.stamp==data['odom'].header.stamp==data['tf'].transforms[0].header.stamp==stamp
    assert list(data['joints'].position)==[.123456789,-.765432109]
    assert list(data['joints'].velocity)==[.234567891,-.345678912]
    assert data['odom'].pose.pose.position.x==pytest.approx(0.)
    assert abs(data['odom'].pose.pose.orientation.w)==pytest.approx(1.)


def test_large_turn_preserves_initial_odometry_frame_and_body_twist():
    item,events=feedback([1.,.2,-.002],2.9)
    item.on_step(.001)
    data=dict(events);odom=data['odom']
    assert item.error==''
    np.testing.assert_allclose([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z],
                               [1.,.2,-.002],atol=1e-12)
    q=odom.pose.pose.orientation
    assert (Rotation.from_quat([q.x,q.y,q.z,q.w]).inv()*Rotation.from_euler('z',2.9)).magnitude()<1e-12
    assert odom.twist.twist.linear.x==pytest.approx(.4)
    assert odom.twist.twist.linear.y==pytest.approx(0.,abs=1e-12)
    assert odom.twist.twist.angular.z==pytest.approx(.8)
    assert odom.header.frame_id=='odom' and odom.child_frame_id=='base_link'


def test_pending_native_command_executes_during_a_physics_step_without_a_render_tick():
    item = runner.PhysicsCommandFilter.__new__(runner.PhysicsCommandFilter)
    item.filter = JointCommandFilter(np.zeros(6), -np.ones(6), np.ones(6), np.ones(6), 0.)
    item.time, item.steps, item.error, item.maximum_command_latency = 0., 0, '', 0.
    item.np, item.indices = np, np.arange(8)
    item.action = lambda **kwargs: SimpleNamespace(**kwargs)
    events = []
    item.robot = SimpleNamespace(apply_action=lambda action: events.append(('applied', action)))
    command = JointState()
    command.name = runner.WHEEL_JOINTS+runner.ARM_JOINTS
    command.position = [float('nan'), float('nan')]+[.2]*6
    command.velocity = [1., -1.]+[.4]*6
    def receive():
        events.append(('received', None))
        item.on_command(command)
    item.reset_server = SimpleNamespace(spin_once=receive)
    item.on_step(.001)
    assert item.error == '' and item.steps == 1
    assert [kind for kind, _ in events] == ['received', 'applied']
    action = events[-1][1]
    np.testing.assert_array_equal(action.joint_positions, [0., 0.]+[.2]*6)
    np.testing.assert_array_equal(action.joint_velocities, [1., -1.]+[.4]*6)
    assert item.maximum_command_latency == pytest.approx(.001)
