"""ROS 2 Bridge Node.

Wraps rclpy functionality and provides async-compatible interface
for the MCP server to interact with ROS 2 topics and services.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

# ROS 2 imports - these will fail if ROS 2 is not sourced
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist, PoseStamped, Pose2D
    from sensor_msgs.msg import LaserScan, Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Mock classes for development without ROS 2
    class Node:  # type: ignore
        pass

logger = structlog.get_logger()


@dataclass
class RobotState:
    """Current state of the robot."""
    
    pose_x: float = 0.0
    pose_y: float = 0.0
    pose_theta: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    scan_ranges: list[float] = field(default_factory=list)
    scan_angle_min: float = 0.0
    scan_angle_max: float = 0.0
    scan_angle_increment: float = 0.0
    last_image: bytes | None = None
    is_moving: bool = False


class ROS2BridgeNode(Node):
    """ROS 2 Node that interfaces with robot topics and services."""

    def __init__(self) -> None:
        super().__init__("mcp_ros2_bridge")
        self.state = RobotState()
        self._setup_publishers()
        self._setup_subscribers()
        self.get_logger().info("MCP-ROS2 Bridge Node initialized")

    def _setup_publishers(self) -> None:
        """Set up ROS 2 publishers."""
        # Velocity command publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )
        self.get_logger().info("Publisher created: /cmd_vel")

    def _setup_subscribers(self) -> None:
        """Set up ROS 2 subscribers."""
        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Odometry subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            10
        )

        # LaserScan subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self._scan_callback,
            sensor_qos
        )

        # Camera subscriber (optional)
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            sensor_qos
        )

        self.get_logger().info("Subscribers created: /odom, /scan, /camera/image_raw")

    def _odom_callback(self, msg: Odometry) -> None:
        """Process odometry data."""
        self.state.pose_x = msg.pose.pose.position.x
        self.state.pose_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.state.pose_theta = float(asyncio.compat.atan2(siny_cosp, cosy_cosp))
        
        self.state.linear_velocity = msg.twist.twist.linear.x
        self.state.angular_velocity = msg.twist.twist.angular.z
        self.state.is_moving = abs(self.state.linear_velocity) > 0.01 or abs(self.state.angular_velocity) > 0.01

    def _scan_callback(self, msg: LaserScan) -> None:
        """Process laser scan data."""
        self.state.scan_ranges = list(msg.ranges)
        self.state.scan_angle_min = msg.angle_min
        self.state.scan_angle_max = msg.angle_max
        self.state.scan_angle_increment = msg.angle_increment

    def _image_callback(self, msg: Image) -> None:
        """Process camera image."""
        self.state.last_image = bytes(msg.data)

    def publish_velocity(self, linear_x: float, angular_z: float) -> None:
        """Publish velocity command."""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(f"Published velocity: linear={linear_x}, angular={angular_z}")

    def stop(self) -> None:
        """Emergency stop - publish zero velocity."""
        self.publish_velocity(0.0, 0.0)
        self.get_logger().info("Emergency stop executed")


class ROS2Bridge:
    """Async wrapper for ROS 2 node, running in a separate thread."""

    def __init__(self) -> None:
        self.node: ROS2BridgeNode | None = None
        self.executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._running = False
        self.is_connected = False

    async def initialize(self) -> None:
        """Initialize ROS 2 context and start spinning in background thread."""
        if not ROS2_AVAILABLE:
            logger.warning("ROS 2 not available - running in mock mode")
            self.is_connected = False
            return

        try:
            # Initialize rclpy if not already done
            if not rclpy.ok():
                rclpy.init()

            # Create node and executor
            self.node = ROS2BridgeNode()
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)

            # Start spinning in background thread
            self._running = True
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()

            self.is_connected = True
            logger.info("ROS 2 bridge initialized and spinning")

        except Exception as e:
            logger.error("Failed to initialize ROS 2 bridge", error=str(e))
            self.is_connected = False
            raise

    def _spin_loop(self) -> None:
        """Spin executor in background thread."""
        while self._running and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.1)

    async def shutdown(self) -> None:
        """Shutdown ROS 2 bridge."""
        self._running = False
        
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

        if self.node:
            self.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        self.is_connected = False
        logger.info("ROS 2 bridge shutdown complete")

    @property
    def state(self) -> RobotState:
        """Get current robot state."""
        if self.node:
            return self.node.state
        return RobotState()

    async def move(self, linear_x: float, angular_z: float, duration_ms: int) -> dict[str, Any]:
        """Execute movement command for specified duration."""
        if not self.node:
            return {"success": False, "error": "ROS 2 not connected"}

        try:
            self.node.publish_velocity(linear_x, angular_z)
            await asyncio.sleep(duration_ms / 1000.0)
            self.node.stop()
            
            return {
                "success": True,
                "linear_x": linear_x,
                "angular_z": angular_z,
                "duration_ms": duration_ms,
                "final_pose": {
                    "x": self.state.pose_x,
                    "y": self.state.pose_y,
                    "theta": self.state.pose_theta,
                }
            }
        except Exception as e:
            logger.error("Move command failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def stop(self) -> dict[str, Any]:
        """Emergency stop."""
        if not self.node:
            return {"success": False, "error": "ROS 2 not connected"}

        self.node.stop()
        return {"success": True, "message": "Emergency stop executed"}

    async def set_velocity(self, linear_x: float, angular_z: float) -> dict[str, Any]:
        """Set continuous velocity (use with caution)."""
        if not self.node:
            return {"success": False, "error": "ROS 2 not connected"}

        self.node.publish_velocity(linear_x, angular_z)
        return {
            "success": True,
            "linear_x": linear_x,
            "angular_z": angular_z,
            "warning": "Continuous velocity set - remember to call stop()"
        }

