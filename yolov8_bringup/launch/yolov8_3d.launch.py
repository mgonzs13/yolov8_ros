# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    #
    # ARGS
    #
    model = LaunchConfiguration("model")
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="yolov8m.pt",
        description="Model name or path")

    tracker = LaunchConfiguration("tracker")
    tracker_cmd = DeclareLaunchArgument(
        "tracker",
        default_value="bytetrack.yaml",
        description="Tracker name or path")

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to use (GPU/CPU)")

    enable = LaunchConfiguration("enable")
    enable_cmd = DeclareLaunchArgument(
        "enable",
        default_value="True",
        description="Whether to start YOLOv8 enabled")

    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.5",
        description="Minimum probability of a detection to be published")

    input_image_topic = LaunchConfiguration("input_image_topic")
    input_image_topic_cmd = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/camera/rgb/image_raw",
        description="Name of the input image topic")

    input_depth_topic = LaunchConfiguration("input_depth_topic")
    input_depth_topic_cmd = DeclareLaunchArgument(
        "input_depth_topic",
        default_value="/camera/depth/image_raw",
        description="Name of the input depth topic")

    input_depth_info_topic = LaunchConfiguration("input_depth_info_topic")
    input_depth_info_topic_cmd = DeclareLaunchArgument(
        "input_depth_info_topic",
        default_value="/camera/depth/camera_info",
        description="Name of the input depth info topic")

    depth_image_units_divisor = LaunchConfiguration(
        "depth_image_units_divisor")
    depth_image_units_divisor_cmd = DeclareLaunchArgument(
        "depth_image_units_divisor",
        default_value="1000",
        description="Divisor used to convert the raw depth image values into metres")

    target_frame = LaunchConfiguration("target_frame")
    target_frame_cmd = DeclareLaunchArgument(
        "target_frame",
        default_value="base_link",
        description="Target frame to transform the 3D boxes")

    maximum_detection_threshold = LaunchConfiguration(
        "maximum_detection_threshold")
    maximum_detection_threshold_cmd = DeclareLaunchArgument(
        "maximum_detection_threshold",
        default_value="0.3",
        description="Maximum detection threshold in the z axis")

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes")

    #
    # NODES
    #
    detector_node_cmd = Node(
        package="yolov8_ros",
        executable="yolov8_node",
        name="yolov8_node",
        namespace=namespace,
        parameters=[{"model": model,
                     "device": device,
                     "enable": enable,
                     "threshold": threshold}],
        remappings=[("image_raw", input_image_topic)]
    )

    tracking_node_cmd = Node(
        package="yolov8_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=namespace,
        parameters=[{"tracker": tracker}],
        remappings=[("image_raw", input_image_topic)]
    )

    detect_3d_node_cmd = Node(
        package="yolov8_ros",
        executable="detect_3d_node",
        name="detect_3d_node",
        namespace=namespace,
        parameters=[{"target_frame": target_frame,
                     "maximum_detection_threshold": maximum_detection_threshold,
                     "depth_image_units_divisor": depth_image_units_divisor}],
        remappings=[("depth_image", input_depth_topic),
                    ("depth_info", input_depth_info_topic),
                    ("detections", "tracking")]
    )

    debug_node_cmd = Node(
        package="yolov8_ros",
        executable="debug_node",
        name="debug_node",
        namespace=namespace,
        remappings=[("image_raw", input_image_topic),
                    ("detections", "detections_3d")]
    )

    ld = LaunchDescription()

    ld.add_action(model_cmd)
    ld.add_action(tracker_cmd)
    ld.add_action(device_cmd)
    ld.add_action(enable_cmd)
    ld.add_action(threshold_cmd)
    ld.add_action(input_image_topic_cmd)
    ld.add_action(input_depth_topic_cmd)
    ld.add_action(input_depth_info_topic_cmd)
    ld.add_action(depth_image_units_divisor_cmd)
    ld.add_action(target_frame_cmd)
    ld.add_action(maximum_detection_threshold_cmd)
    ld.add_action(namespace_cmd)

    ld.add_action(detector_node_cmd)
    ld.add_action(tracking_node_cmd)
    ld.add_action(detect_3d_node_cmd)
    ld.add_action(debug_node_cmd)

    return ld
