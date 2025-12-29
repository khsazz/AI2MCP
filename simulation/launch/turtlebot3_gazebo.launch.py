"""Launch file for TurtleBot3 in Gazebo with MCP bridge.

This launch file sets up:
1. Gazebo simulation with TurtleBot3
2. ROS 2 robot state publisher
3. (Optional) Nav2 navigation stack
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for TurtleBot3 Gazebo simulation."""
    
    # Get package directories
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world', default='turtlebot3_world')
    robot_model = LaunchConfiguration('model', default='burger')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    enable_nav2 = LaunchConfiguration('enable_nav2', default='false')
    
    # Set TurtleBot3 model environment variable
    set_tb3_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value=robot_model
    )
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    declare_world = DeclareLaunchArgument(
        'world',
        default_value='turtlebot3_world',
        description='Gazebo world file name'
    )
    
    declare_model = DeclareLaunchArgument(
        'model',
        default_value='burger',
        choices=['burger', 'waffle', 'waffle_pi'],
        description='TurtleBot3 model type'
    )
    
    declare_x_pose = DeclareLaunchArgument(
        'x_pose',
        default_value='0.0',
        description='Initial X position'
    )
    
    declare_y_pose = DeclareLaunchArgument(
        'y_pose',
        default_value='0.0',
        description='Initial Y position'
    )
    
    declare_enable_nav2 = DeclareLaunchArgument(
        'enable_nav2',
        default_value='false',
        description='Enable Nav2 navigation stack'
    )
    
    # Include TurtleBot3 Gazebo launch
    turtlebot3_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(turtlebot3_gazebo_dir, 'launch', 'turtlebot3_world.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'x_pose': x_pose,
            'y_pose': y_pose,
        }.items(),
    )
    
    # Static transform for camera (if not provided by robot model)
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'camera_link'],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    return LaunchDescription([
        # Environment
        set_tb3_model,
        
        # Arguments
        declare_use_sim_time,
        declare_world,
        declare_model,
        declare_x_pose,
        declare_y_pose,
        declare_enable_nav2,
        
        # Launch TurtleBot3 in Gazebo
        turtlebot3_gazebo_launch,
        
        # Additional transforms
        static_tf_camera,
    ])

