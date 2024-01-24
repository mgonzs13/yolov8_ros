# Copyright (C) 2024  Miguel Ángel González Santamarta

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


import copy
import math
from threading import Lock

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray


class SpeedEstimateNode(Node):

    def __init__(self) -> None:
        super().__init__("speed_estimate_node")

        self.pd_lock = Lock()
        self.previous_detections = {}

        self._pub = self.create_publisher(
            DetectionArray, "detections_speed", 10)
        self._sub = self.create_subscription(
            DetectionArray, "detections_3d", self.detections_cb, 10)

        self._timer = self.create_timer(0.25, self.clean_detections)

    def calculate_euclidian(self, old_pos: Pose, cur_pos: Pose) -> float:
        return math.sqrt(
            math.pow(old_pos.position.x - cur_pos.position.x, 2) +
            math.pow(old_pos.position.y - cur_pos.position.y, 2) +
            math.pow(old_pos.position.z - cur_pos.position.z, 2)
        )

    def detections_cb(self, detection_msg: DetectionArray) -> None:

        timestamp = detection_msg.header.stamp.sec + \
            detection_msg.header.stamp.nanosec / 1e9

        detection: Detection
        for detection in detection_msg.detections:

            if detection.id in self.previous_detections:

                old_pos = self.previous_detections[detection.id]["position"]
                cur_pos = detection.bbox3d.center

                dist = self.calculate_euclidian(old_pos, cur_pos)
                t = timestamp - \
                    self.previous_detections[detection.id]["timestamp"]

                detection.speed = dist / t

            else:
                detection.speed = math.nan

            self.previous_detections[detection.id] = {
                "timestamp": timestamp,
                "position": detection.bbox3d.center
            }

        self._pub.publish(detection_msg)

    def clean_detections(self) -> None:

        cur_time = self.get_clock().now().nanoseconds / 1e9

        detections = copy.deepcopy(self.previous_detections)
        for detection_id in detections:

            if cur_time - self.previous_detections[detection_id]["timestamp"] > 1:

                del self.previous_detections[detection_id]


def main():
    rclpy.init()
    node = SpeedEstimateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
