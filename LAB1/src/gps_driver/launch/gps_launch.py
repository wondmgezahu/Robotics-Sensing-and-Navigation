from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'port',
            default_value='/dev/ttyUSB0',
            description='Serial port for GPS'
        ),
        Node(
            package='gps_driver',
            executable='driver',
            name='driver',
            parameters=[{'port': LaunchConfiguration('port')}]
        )
    ])
