"""Launch file for MCP-ROS2 Bridge.

Launches the MCP server that exposes ROS 2 topics as MCP tools/resources.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for MCP bridge."""
    
    # Launch arguments
    host = LaunchConfiguration('host', default='0.0.0.0')
    port = LaunchConfiguration('port', default='8080')
    log_level = LaunchConfiguration('log_level', default='info')
    
    declare_host = DeclareLaunchArgument(
        'host',
        default_value='0.0.0.0',
        description='MCP server host'
    )
    
    declare_port = DeclareLaunchArgument(
        'port',
        default_value='8080',
        description='MCP server port'
    )
    
    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        choices=['debug', 'info', 'warning', 'error'],
        description='Logging level'
    )
    
    # Start MCP server as a process
    # Note: In production, this would be a proper ROS 2 node
    mcp_server = ExecuteProcess(
        cmd=[
            'python3', '-m', 'mcp_ros2_bridge.server',
        ],
        name='mcp_ros2_bridge',
        output='screen',
        shell=True,
    )
    
    return LaunchDescription([
        declare_host,
        declare_port,
        declare_log_level,
        mcp_server,
    ])

