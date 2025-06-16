#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <Eigen/Geometry>
#include <chrono>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <control_msgs/msg/joint_jog.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <sensor_msgs/msg/joint_state.hpp>

// 使用新的 moveit_servo 头文件
#include <moveit_servo/servo.hpp>
#include <moveit_servo/utils/datatypes.hpp>
#include <moveit_servo/utils/command.hpp>

#include <deque>
#include <mutex>

enum class ControlMode { INCREMENTAL, ABSOLUTE };

class VR2ServoNode : public rclcpp::Node
{
private:
    using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
    using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_traj_pub_;
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr traj_action_client_;
    rclcpp::TimerBase::SharedPtr home_timer_;

    // MoveIt相关
    std::shared_ptr<moveit::core::RobotModel> robot_model_;
    std::shared_ptr<moveit::core::RobotState> robot_state_;
    std::string tip_link_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    // Servo相关
    std::shared_ptr<moveit_servo::Servo::ParamListener> servo_param_listener_;
    moveit_servo::Servo::Params servo_params_;
    std::shared_ptr<moveit_servo::Servo> servo_;
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
    const moveit::core::JointModelGroup* joint_model_group_{nullptr};
    std::deque<moveit_servo::KinematicState> joint_cmd_rolling_window_;
    std::mutex pose_guard_;

    ControlMode mode_;
    double alpha_; // 滤波系数
    std::vector<double> last_input_{0,0,0,0,0,0};
    std::vector<double> filtered_input_{0,0,0,0,0,0};
    std::vector<double> initial_pose_{};
    bool first_msg_{true};
    bool ready_;

    std::vector<std::string> joint_names_{"joint1", "joint2", "joint3", 
                                        "joint4", "joint5", "joint6"}; 

    geometry_msgs::msg::Pose get_current_end_effector_pose()
    {
        geometry_msgs::msg::Pose pose_msg;
        if (!robot_state_ || !robot_model_) return pose_msg;
        const Eigen::Isometry3d& end_effector_state = robot_state_->getGlobalLinkTransform(tip_link_);
        pose_msg.position.x = end_effector_state.translation().x();
        pose_msg.position.y = end_effector_state.translation().y();
        pose_msg.position.z = end_effector_state.translation().z();
        Eigen::Quaterniond q(end_effector_state.rotation());
        pose_msg.orientation.x = q.x();
        pose_msg.orientation.y = q.y();
        pose_msg.orientation.z = q.z();
        pose_msg.orientation.w = q.w();
        return pose_msg;
    }

    void process_target_pose_servo(const geometry_msgs::msg::Pose& current_pose, 
                             const geometry_msgs::msg::Pose& target_pose)
    {
        // 1. 计算位置误差
        Eigen::Vector3d pos_error(
            target_pose.position.x - current_pose.position.x,
            target_pose.position.y - current_pose.position.y,
            target_pose.position.z - current_pose.position.z
        );

        // 2. 计算线速度
        double max_linear_speed = 0.2;
        Eigen::Vector3d linear_vel;
        if (pos_error.norm() > 1e-6) {
            linear_vel = pos_error.normalized() * max_linear_speed * pos_error.norm();
        } else {
            linear_vel = Eigen::Vector3d::Zero();
        }

        // 3. 计算姿态误差（四元数差转角速度）
        tf2::Quaternion q_current, q_target;
        tf2::fromMsg(current_pose.orientation, q_current);
        tf2::fromMsg(target_pose.orientation, q_target);
        tf2::Quaternion q_diff = q_target * q_current.inverse();
        tf2::Vector3 angular_vel_axis = q_diff.getAxis();
        double angular_vel_magnitude = q_diff.getAngle();
        double max_angular_speed = 0.5;
        Eigen::Vector3d angular_vel =
            Eigen::Vector3d(angular_vel_axis.x(), angular_vel_axis.y(), angular_vel_axis.z()) *
            max_angular_speed * angular_vel_magnitude;

        // 4. 组装TwistCommand
        moveit_servo::TwistCommand target_twist{
            servo_params_.planning_frame,
            { linear_vel.x(), linear_vel.y(), linear_vel.z(), angular_vel.x(), angular_vel.y(), angular_vel.z() }
        };

        // 5. servo计算下一步关节状态
        auto joint_state = servo_->getNextJointState(robot_state_, target_twist);
        auto status = servo_->getStatus();

        // 修正数据类型转换
        if (status != moveit_servo::StatusCode::INVALID)
        {
            auto goal_msg = std::make_shared<FollowJointTrajectory::Goal>();
            goal_msg->trajectory.joint_names = joint_names_;
            
            trajectory_msgs::msg::JointTrajectoryPoint point;
            // 将 Eigen::VectorXd 转换为 std::vector
            point.positions.resize(joint_state.positions.size());
            point.velocities.resize(joint_state.velocities.size());
            Eigen::VectorXd::Map(&point.positions[0], joint_state.positions.size()) = joint_state.positions;
            Eigen::VectorXd::Map(&point.velocities[0], joint_state.velocities.size()) = joint_state.velocities;
            point.time_from_start = rclcpp::Duration::from_seconds(0.1);
            goal_msg->trajectory.points.push_back(point);
            
            // 发送到 rm_control
            auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();
            send_goal_options.goal_response_callback =
                [this](const auto& goal_handle) {
                    if (!goal_handle) {
                        RCLCPP_ERROR(this->get_logger(), "Goal rejected");
                    }
                };
                
            traj_action_client_->async_send_goal(*goal_msg, send_goal_options);
            
            // 更新robot_state
            robot_state_->setJointGroupPositions(joint_model_group_, joint_state.positions);
            robot_state_->setJointGroupVelocities(joint_model_group_, joint_state.velocities);
        }
    }

    void chatter_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 6) return;

        // --- 滤波 ---
        if (first_msg_) {
            filtered_input_ = std::vector<double>(msg->data.begin(), msg->data.begin()+6);
            initial_pose_ = filtered_input_;
            first_msg_ = false;
        } else {
            for (size_t i = 0; i < 6; ++i) {
                filtered_input_[i] = alpha_ * msg->data[i] + (1 - alpha_) * filtered_input_[i];
            }
        }

        if (mode_ == ControlMode::ABSOLUTE) {
            geometry_msgs::msg::Pose target_pose;
            target_pose.position.x = filtered_input_[0] - initial_pose_[0];
            target_pose.position.y = filtered_input_[1] - initial_pose_[1];
            target_pose.position.z = filtered_input_[2] - initial_pose_[2];

            double pitch = filtered_input_[3] - initial_pose_[3];
            double yaw   = filtered_input_[4] - initial_pose_[4];
            double roll  = filtered_input_[5] - initial_pose_[5];
            Eigen::AngleAxisd rollAngle(roll,   Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
            Eigen::AngleAxisd yawAngle(yaw,     Eigen::Vector3d::UnitZ());
            Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
            target_pose.orientation.x = q.x();
            target_pose.orientation.y = q.y();
            target_pose.orientation.z = q.z();
            target_pose.orientation.w = q.w();

            // 直接处理目标位姿
            geometry_msgs::msg::Pose current_pose = get_current_end_effector_pose();
            process_target_pose_servo(current_pose, target_pose);
        } else {
            // 增量模式：输入为速度，直接发Twist
            geometry_msgs::msg::TwistStamped twist;
            twist.header.stamp = now();
            twist.header.frame_id = "dummy_link";
            twist.twist.linear.x  = filtered_input_[0];
            twist.twist.linear.y  = filtered_input_[1];
            twist.twist.linear.z  = filtered_input_[2];
            twist.twist.angular.x = filtered_input_[3] * 0.0;
            twist.twist.angular.y = filtered_input_[4] * 0.0;
            twist.twist.angular.z = filtered_input_[5] * 0.0;
            twist_pub_->publish(twist);
        }
    }

public:
    VR2ServoNode()
    : Node("vr2servo_node"),
      mode_(ControlMode::INCREMENTAL),
      alpha_(0.2),
      ready_(true)
    {
        declare_parameter("mode", "absolute"); // "incremental" or "absolute"
        declare_parameter("alpha", 0.2);
        std::string mode_param = get_parameter("mode").as_string();
        alpha_ = get_parameter("alpha").as_double();
        if (mode_param == "absolute") mode_ = ControlMode::ABSOLUTE;

        sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            "/chatter", 10,
            std::bind(&VR2ServoNode::chatter_callback, this, std::placeholders::_1)
        );
        twist_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
            "/rm_servo/delta_twist_cmds", 10
        );
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/rm_servo/target_pose", 10
        );
        joint_traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/rm_group_controller/joint_trajectory", 10
        );
        traj_action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this, "/rm_group_controller/follow_joint_trajectory");

        // MoveIt相关初始化
        robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
        robot_model_ = robot_model_loader.getModel();
        robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
        robot_state_->setToDefaultValues();
        tip_link_ = "Link8"; // 请根据实际机械臂末端link名修改

        joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                robot_state_->setVariablePositions(msg->name, msg->position);
            }
        );

        // MoveIt Servo 初始化
        servo_param_listener_ = std::make_shared<moveit_servo::servo::ParamListener>(shared_from_this(), "moveit_servo");
        servo_params_ = servo_param_listener_->get_params();
        planning_scene_monitor_ = moveit_servo::createPlanningSceneMonitor(shared_from_this(), servo_params_);
        
        // 创建 servo 实例
        servo_ = std::make_shared<moveit_servo::Servo>(
            shared_from_this(),
            servo_param_listener_,
            planning_scene_monitor_
        );
        servo_->setCommandType(moveit_servo::CommandType::TWIST);
        joint_model_group_ = robot_state_->getJointModelGroup(servo_params_.move_group_name);

        RCLCPP_INFO(get_logger(), "vr2servo_node started, mode: %s", mode_param.c_str());

        // 注释掉自动回零
        // this->wait_and_send_home();
    }

    ~VR2ServoNode()
    {
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VR2ServoNode>());
    rclcpp::shutdown();
    return 0;
}