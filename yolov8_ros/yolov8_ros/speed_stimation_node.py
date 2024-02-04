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
from threading import Lock

import numpy as np
from filterpy.kalman import KalmanFilter

import rclpy
from rclpy.node import Node

from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray


class SpeedEstimateNode(Node):

    def __init__(self) -> None:
        super().__init__("speed_estimate_node")

        self.pd_lock = Lock()
        self.previous_detections = {}

        self.declare_parameter("use_kalman", True)
        self.use_kalman = self.get_parameter(
            "use_kalman").get_parameter_value().bool_value

        self._pub = self.create_publisher(
            DetectionArray, "detections_speed", 10)
        self._sub = self.create_subscription(
            DetectionArray, "detections_3d", self.detections_cb, 10)

        self._timer = self.create_timer(0.25, self.clean_detections)

    def detections_cb(self, detection_msg: DetectionArray) -> None:

        timestamp = detection_msg.header.stamp.sec + \
            detection_msg.header.stamp.nanosec / 1e9

        detection: Detection
        for detection in detection_msg.detections:

            cur_pos = detection.bbox3d.center

            if detection.id in self.previous_detections:

                t = timestamp - \
                    self.previous_detections[detection.id]["timestamp"]

                if not self.use_kalman:
                    old_pos = self.previous_detections[detection.id]["position"]
                    detection.velocity.linear.x = (
                        cur_pos.position.x - old_pos.position.x) / t
                    detection.velocity.linear.y = (
                        cur_pos.position.y - old_pos.position.y) / t
                    detection.velocity.linear.z = (
                        cur_pos.position.z - old_pos.position.z) / t

                else:
                    kf: KalmanFilter = self.previous_detections[detection.id]["kf"]
                    self.update_kf(kf, t)
                    kf.predict()
                    kf.update(np.array([cur_pos.position.x,
                                        cur_pos.position.y,
                                        cur_pos.position.z]))
                    detection.velocity.linear.x = kf.x[3]
                    detection.velocity.linear.y = kf.x[4]
                    detection.velocity.linear.z = kf.x[5]
            else:
                detection.velocity.linear.x = np.nan
                detection.velocity.linear.y = np.nan
                detection.velocity.linear.z = np.nan

            if detection.id not in self.previous_detections:
                self.previous_detections[detection.id] = {}

            self.previous_detections[detection.id]["timestamp"] = timestamp
            self.previous_detections[detection.id]["position"] = detection.bbox3d.center

            if self.use_kalman and "kf" not in self.previous_detections[detection.id]:
                self.previous_detections[detection.id]["kf"] = self.init_kf(np.array([cur_pos.position.x,
                                                                                      cur_pos.position.y,
                                                                                      cur_pos.position.z]))

        self._pub.publish(detection_msg)

    def clean_detections(self) -> None:

        cur_time = self.get_clock().now().nanoseconds / 1e9

        detections = copy.deepcopy(self.previous_detections)
        for detection_id in detections:

            if cur_time - self.previous_detections[detection_id]["timestamp"] > 1:

                del self.previous_detections[detection_id]

    def init_kf(self, initial_pose: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=9, dim_z=3)

        # initial state [x, y, z, vx, vy, vz, ax, ay, az]
        kf.x = np.zeros(9)
        kf.x[0] = initial_pose[0]
        kf.x[1] = initial_pose[1]
        kf.x[2] = initial_pose[2]

        # state transition matrix
        kf.F = np.eye(9)

        # measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        # uncertainty covariance
        kf.P = np.eye(9) * 1e3

        # measurement noise covariance matrix R
        kf.R = np.eye(3) * 0.1

        # process noise covariance matrix Q
        kf.Q = np.eye(9) * 0.01

        return kf

    def update_kf(self, kf: KalmanFilter, dt: float) -> None:
        # update the state transition matrix F
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])


def main():
    rclpy.init()
    node = SpeedEstimateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
