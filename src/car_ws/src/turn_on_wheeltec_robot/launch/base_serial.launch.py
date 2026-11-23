from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    akmcar = LaunchConfiguration('akmcar', default='true')

    odom_x_scale_arg = DeclareLaunchArgument(
        'odom_x_scale',
        default_value='1.0',
        description='Odometry X linear distance calibration scale factor',
    )
    odom_y_scale_arg = DeclareLaunchArgument(
        'odom_y_scale',
        default_value='1.0',
        description='Odometry Y linear distance calibration scale factor',
    )
    odom_z_scale_pos_arg = DeclareLaunchArgument(
        'odom_z_scale_positive',
        default_value='1.0',
        description='Odometry positive Z angular calibration scale factor',
    )
    odom_z_scale_neg_arg = DeclareLaunchArgument(
        'odom_z_scale_negative',
        default_value='1.0',
        description='Odometry negative Z angular calibration scale factor',
    )

    return LaunchDescription([
        odom_x_scale_arg,
        odom_y_scale_arg,
        odom_z_scale_pos_arg,
        odom_z_scale_neg_arg,
        launch_ros.actions.Node(
            package='turn_on_wheeltec_robot',
            executable='wheeltec_robot_node',
            output='screen',
            parameters=[{
                'usart_port_name': '/dev/wheeltec_controller',
                'serial_baud_rate': 115200,
                'robot_frame_id': 'base_footprint',
                'odom_frame_id': 'odom_combined',
                'cmd_vel': 'cmd_vel',
                'akm_cmd_vel': 'none',
                'product_number': 0,
                'odom_x_scale': ParameterValue(
                    LaunchConfiguration('odom_x_scale'), value_type=float
                ),
                'odom_y_scale': ParameterValue(
                    LaunchConfiguration('odom_y_scale'), value_type=float
                ),
                'odom_z_scale_positive': ParameterValue(
                    LaunchConfiguration('odom_z_scale_positive'), value_type=float
                ),
                'odom_z_scale_negative': ParameterValue(
                    LaunchConfiguration('odom_z_scale_negative'), value_type=float
                ),
            }],
        ),
    ])
