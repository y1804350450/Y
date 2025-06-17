from reader import OculusReader
from transforms3d.quaternions import mat2quat as quaternion_from_matrix
import rclpy
import numpy as np
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_matrix, quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Pose


class OculusVisualizer(Node):
    def __init__(self):
        super().__init__('oculus_visualizer')
        self.oculus_reader = OculusReader()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        # 声明 oculus_base 的参数
        self.declare_parameter('oculus_base.x', 0.05)
        self.declare_parameter('oculus_base.y', 0.0)
        self.declare_parameter('oculus_base.z', 0.45)
        self.declare_parameter('oculus_base.roll', 3.14 / 2.0)
        self.declare_parameter('oculus_base.pitch', 0.0)
        self.declare_parameter('oculus_base.yaw', -3.14 / 2)

        # 初始化并发布静态变换
        self.publish_static_transform()

        # 初始化并发布静态变换
        self.publish_static_transform()

        # 初始化 TF2 Buffer 和 Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 创建消息客户端
        self.pose_publisher_ = self.create_publisher(Pose, '/chatter', 10)

        self.scale = 1.5  # translation缩放因子

        #过滤跳变
        self.last_transform = None

    def publish_static_transform(self):
        """发布从 world 到 oculus_base 的静态变换"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'oculus_base'

        x = self.get_parameter('oculus_base.x').get_parameter_value().double_value
        y = self.get_parameter('oculus_base.y').get_parameter_value().double_value
        z = self.get_parameter('oculus_base.z').get_parameter_value().double_value
        roll = self.get_parameter('oculus_base.roll').get_parameter_value().double_value
        pitch = self.get_parameter('oculus_base.pitch').get_parameter_value().double_value
        yaw = self.get_parameter('oculus_base.yaw').get_parameter_value().double_value

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        quat = quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.static_tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Published static transform from world to oculus_base: "
                              f"position=[{x}, {y}, {z}], euler=[{roll}, {pitch}, {yaw}]")

    def publish_transform(self, transform, name):
        try:
            if transform.shape != (4, 4):
                self.get_logger().warning(f"Invalid transform shape for {name}: {transform.shape}")
                return

            rotation_matrix = transform[:3, :3]
            det = np.linalg.det(rotation_matrix)
            if not np.isclose(det, 1.0, atol=1e-5):
                self.get_logger().warning(f"Invalid rotation matrix for {name}: determinant = {det}")
                return

            translation = transform[:3, 3]
            if name == 'oculus_right':
                if self.last_transform is not None:

                    diff = np.linalg.norm(translation - self.last_transform)
                    if diff > 0.1:  # 设定一个阈值
                        self.last_transform=translation
                        return
                else:
                    self.last_transform = translation
            
            
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'oculus_base'
            t.child_frame_id = name
            t.transform.translation.x = float(translation[0])
            t.transform.translation.y = float(translation[1])
            t.transform.translation.z = float(translation[2])

            quat = quaternion_from_matrix(transform)
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])

            self.tf_broadcaster.sendTransform(t)  # 修正这里
        except Exception as e:
            self.get_logger().warning(f"Error publishing transform for {name}: {str(e)}")

    def publish_target_pose(self, transform_stamped):
        """发送末端目标位姿给 set_target_pose 服务"""
        pose = Pose()
        pose.position.x = transform_stamped.transform.translation.x
        pose.position.y = transform_stamped.transform.translation.y
        pose.position.z = transform_stamped.transform.translation.z
        pose.orientation = transform_stamped.transform.rotation

        self.pose_publisher_.publish(pose)
        self.get_logger().info(f"Published target pose: {pose}")

    def timer_callback(self):
        # 获取 Oculus 控制器的变换和按钮状态
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        self.get_logger().info(f'transformations: {transformations}')
        self.get_logger().info(f'buttons: {buttons}')
        # 定义绕 Z 轴 -90 度的旋转矩阵
        rot_z_minus_90 = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # 设置旋转角度（90度转换为弧度）
        y_angle = np.deg2rad(90)  # 或者直接用 np.pi/2

        # 绕 Y 轴旋转 90° 的变换矩阵
        rot_y_90 = np.array([
            [np.cos(y_angle), 0.0, np.sin(y_angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(y_angle), 0.0, np.cos(y_angle), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # # 处理右控制器
        # if 'r' in transformations:
        #     right_controller_pose = transformations['r']
            # right_controller_pose[0:3, 3] *= self.scale  # 缩放位移
            # modified_right_pose = np.dot(right_controller_pose, rot_z_minus_90)
            # modified_right_pose = np.dot(modified_right_pose, rot_y_90)
            # self.publish_transform(modified_right_pose, 'oculus_right')

        # # 处理左控制器
        # if 'l' in transformations:
        #     left_controller_pose = transformations['l']
        #     left_controller_pose[0:3, 3] *= self.scale
        #     modified_left_pose = np.dot(left_controller_pose, rot_z_minus_90)
        #     self.publish_transform(modified_left_pose, 'oculus_left')

        # 处理右控制器(不进行旋转)
        if 'r' in transformations:
            right_controller_pose = transformations['r']
            right_controller_pose[0:3, 3] *= self.scale  # 缩放位移
            self.publish_transform(right_controller_pose, 'oculus_right')
            
        
        # 处理左控制器(不进行旋转)
        if 'l' in transformations:
            left_controller_pose = transformations['l']
            left_controller_pose[0:3, 3] *= self.scale  # 缩放位移
            self.publish_transform(left_controller_pose, 'oculus_left')

        # 处理 rightTrig 右扳机控制末端位姿
        if 'rightTrig' in buttons and buttons['rightTrig'][0] > 0.0:
            try:
                trans = self.tf_buffer.lookup_transform(
                    'world',
                    'oculus_right',
                    rclpy.time.Time()
                )
                self.publish_target_pose(trans)
            except Exception as e:
                self.get_logger().warn(f"Failed to lookup transform: {str(e)}")

        
        


def main():
    rclpy.init()
    node = OculusVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
