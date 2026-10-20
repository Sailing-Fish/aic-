import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("line_follower")
    default_params = os.path.join(package_share, "config", "line_follower_params.yaml")

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/image_raw",
        description="订阅的 RGB 图像话题",
    )

    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="底盘控制输出话题",
    )

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="巡线节点参数文件路径",
    )

    image_topic = LaunchConfiguration("image_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    params_file = LaunchConfiguration("params_file")

    node = Node(
        package="line_follower",
        executable="line_follower_node",
        name="line_follower_node",
        output="screen",
        parameters=[
            params_file,
            {
                "image_topic": image_topic,
                "cmd_vel_topic": cmd_vel_topic,
            },
        ],
    )

    return LaunchDescription(
        [
            image_topic_arg,
            cmd_vel_topic_arg,
            params_arg,
            node,
        ]
    )
