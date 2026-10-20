import os

from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from nav2_common.launch import RewrittenYaml

def generate_robot_node(robot_urdf,child):
    return launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name=f'robot_state_publisher_{child}',
        arguments=[os.path.join(get_package_share_directory('wheeltec_robot_urdf'), 'urdf', robot_urdf)],
    )

def generate_static_transform_publisher_node(translation, rotation, parent, child):
    return launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=f'base_to_{child}',
        arguments=[translation[0], translation[1], translation[2], rotation[0], rotation[1], rotation[2], parent, child],
    )
    
def generate_launch_description():
    mini_akm = LaunchConfiguration('mini_akm', default='false')

    mini_akm_ = GroupAction(
        condition=IfCondition(mini_akm), 
        actions=[
        generate_robot_node('mini_akm_robot.urdf','mini_akm'),
        generate_static_transform_publisher_node(['0.05134 ', '0', '0.09452'], ['0', '0', '0'], 'base_footprint', 'laser'),
        generate_static_transform_publisher_node(['0.11365 ', '0.00017', '0.0762'], ['0', '0', '0'], 'base_footprint', 'camera_link'),    
    ]) 

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add the actions to launch all of the localiztion nodes

    ld.add_action(mini_akm_)
    return ld
