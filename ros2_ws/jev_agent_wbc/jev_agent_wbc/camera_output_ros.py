"""Publish the selected render and matching acquisition-time geometry to ROS."""
import numpy as np


class CameraPublisher:
    def __init__(self, source):
        import rclpy
        from sensor_msgs.msg import Image, CameraInfo
        from geometry_msgs.msg import PoseStamped
        from tf2_ros import TransformBroadcaster
        self.rclpy = rclpy
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init(args=[])
        self.node = rclpy.create_node('cinematic_camera_output')
        self.frame = 'virtual_camera_stabilized' if source == 'stabilized' else 'camera_execution_output'
        self.image = self.node.create_publisher(Image, '/camera/image', 1)
        self.raw = self.node.create_publisher(Image, '/camera/debug/execution/image', 1)
        self.info = self.node.create_publisher(CameraInfo, '/camera/camera_info', 1)
        self.pose = self.node.create_publisher(PoseStamped, '/camera/pose', 1)
        self.tf = TransformBroadcaster(self.node)

    def publish(self, rgb, raw_rgb, record, stamp_s, intrinsics):
        from sensor_msgs.msg import Image, CameraInfo
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from std_msgs.msg import Header
        header = Header()
        stamp_ns = round(stamp_s*1e9)
        header.stamp.sec, header.stamp.nanosec = divmod(stamp_ns, 10**9)
        header.frame_id = self.frame
        def image(values, frame):
            values = np.ascontiguousarray(values, dtype=np.uint8)
            message = Image()
            message.header.stamp = header.stamp
            message.header.frame_id = frame
            message.height, message.width = values.shape[:2]
            message.encoding, message.step, message.data = 'rgb8', values.shape[1]*3, values.tobytes()
            return message
        self.image.publish(image(rgb, self.frame))
        self.raw.publish(image(raw_rgb, 'camera_link'))
        height, width = rgb.shape[:2]
        fx, fy, cx, cy = intrinsics
        info = CameraInfo()
        info.header, info.height, info.width = header, height, width
        info.distortion_model, info.d = 'plumb_bob', [0., 0., 0., 0., 0.]
        info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        info.k = [float(fx*width), 0., float(cx*width), 0., float(fy*height), float(cy*height), 0., 0., 1.]
        info.p = [info.k[0], 0., info.k[2], 0., 0., info.k[4], info.k[5], 0., 0., 0., 1., 0.]
        self.info.publish(info)
        pose = PoseStamped()
        # The existing Isaac odometry adapter names its world Z-up frame odom.
        pose.header.stamp, pose.header.frame_id = header.stamp, 'odom'
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, record['camera_position'])
        q = record['camera_quaternion_xyzw']
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, q)
        self.pose.publish(pose)
        transform = TransformStamped()
        transform.header, transform.child_frame_id = pose.header, self.frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, record['camera_position'])
        transform.transform.rotation = pose.pose.orientation
        self.tf.sendTransform(transform)
        self.rclpy.spin_once(self.node, timeout_sec=0.)

    def close(self):
        self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()
