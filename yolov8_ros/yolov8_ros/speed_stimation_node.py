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


import cv2
import copy
import numpy as np
from typing import List
from threading import Lock

import rclpy
from rclpy.node import Node
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray


class SpeedEstimateNode(Node):

    def __init__(self) -> None:
        super().__init__("speed_estimate_node")

        self.pd_lock = Lock()
        self.previous_detections = {}
        self.noise_cov = [100., 2., 2.]

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
                    kf: cv2.KalmanFilter = self.previous_detections[detection.id]["kf"]
                    self.update_kf(kf, t)
                    kf.predict()
                    measurement = np.array(
                        [[cur_pos.position.x, cur_pos.position.y, cur_pos.position.z]]).T.astype(np.float32)
                    kf.correct(measurement)
                    detection.velocity.linear.x = np.float(kf.statePost[3])
                    detection.velocity.linear.y = np.float(kf.statePost[4])
                    detection.velocity.linear.z = np.float(kf.statePost[5])
            else:
                detection.velocity.linear.x = np.nan
                detection.velocity.linear.y = np.nan
                detection.velocity.linear.z = np.nan

            if detection.id not in self.previous_detections:
                self.previous_detections[detection.id] = {}

            self.previous_detections[detection.id]["timestamp"] = timestamp
            self.previous_detections[detection.id]["position"] = detection.bbox3d.center

            if self.use_kalman and "kf" not in self.previous_detections[detection.id]:
                self.previous_detections[detection.id]["kf"] = self.init_kf([
                    cur_pos.position.x,
                    cur_pos.position.y,
                    cur_pos.position.z
                ])

        self._pub.publish(detection_msg)

    def clean_detections(self) -> None:

        cur_time = self.get_clock().now().nanoseconds / 1e9

        detections = copy.deepcopy(list(self.previous_detections.keys()))
        for detection_id in detections:

            if cur_time - self.previous_detections[detection_id]["timestamp"] > 1:

                del self.previous_detections[detection_id]

    def init_kf(self, initial_pose: List[float]) -> cv2.KalmanFilter:

        kf = cv2.KalmanFilter(6, 3)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        kf.measurementNoiseCov = np.eye(3).astype(np.float32)

        kf.statePost = np.concatenate(
            [initial_pose, [0, 0, 0]]
        ).astype(np.float32)

        kf.errorCovPost = np.diag(
            [1., 1., 1., 10., 10., 10.]
        ).astype(np.float32)

        return kf

    def update_kf(self, kf: cv2.KalmanFilter, dt: float) -> None:
        # update the state transition matrix F
        kf.transitionMatrix = np.eye(6).astype(np.float32)
        kf.transitionMatrix[0, 3] = dt
        kf.transitionMatrix[1, 4] = dt
        kf.transitionMatrix[2, 5] = dt

        # update the Q matrix
        # https://github.com/ros-planning/navigation2_dynamic/blob/6eae17e563ec28b9bc9f658748c4aa82e3e0fd2a/kf_hungarian_tracker/kf_hungarian_tracker/obstacle_class.py#L64
        dt2 = dt**2
        dt3 = dt * dt2
        dt4 = dt2**2
        kf.processNoiseCov = np.array([
            [dt4*self.noise_cov[0]/4, 0, 0, dt3*self.noise_cov[0]/2, 0, 0],
            [0, dt4*self.noise_cov[1]/4, 0, 0, dt3*self.noise_cov[1]/2, 0],
            [0, 0, dt4*self.noise_cov[2]/4, 0, 0, dt3*self.noise_cov[2]/2],
            [dt3*self.noise_cov[0]/2, 0, 0, dt2*self.noise_cov[0], 0, 0],
            [0, dt3*self.noise_cov[1]/2, 0, 0, dt2*self.noise_cov[1], 0],
            [0, 0, dt3*self.noise_cov[2]/2, 0, 0, dt2*self.noise_cov[2]]
        ]).astype(np.float32)


def main():
    rclpy.init()
    node = SpeedEstimateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
