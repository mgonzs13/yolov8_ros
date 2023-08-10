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


import numpy as np
from typing import Tuple

from sklearn.cluster import KMeans

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from yolov8_msgs.msg import BoundingBox3D


class BBox3DNode(Node):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.target_frame = self.get_parameter(
            "target_frame").get_parameter_value().string_value

        # aux
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pub
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

        # subscribers
        self.points_sub = message_filters.Subscriber(
            self, PointCloud2, "points",
            qos_profile=qos_profile_sensor_data)
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.points_sub, self.detections_sub), 1, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

    def transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from pointcloud frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                rclpy.time.Time())

            translation = np.array([transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    transform.transform.translation.z])

            rotation = np.array([transform.transform.rotation.w,
                                 transform.transform.rotation.x,
                                 transform.transform.rotation.y,
                                 transform.transform.rotation.z])

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    def on_detections(self,
                      points_msg: PointCloud2,
                      detections_msg: DetectionArray,
                      ) -> None:

        # check if there are detections
        if not detections_msg.detections:
            return

        transform = self.transform(points_msg.header.frame_id)

        if transform is None:
            return

        points = np.frombuffer(points_msg.data, np.float32).reshape(
            points_msg.height, points_msg.width, -1)[:, :, :3]

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header

        for detection in detections_msg.detections:
            bb3d = self.convert_to_3d(points, detection)

            if bb3d is not None:
                new_detections_msg.detections.append(detection)
                new_detections_msg.detections[-1].box3d = BBox3DNode.transform_3d_box(
                    bb3d, transform[0], transform[1])

        self._pub.publish(new_detections_msg)

    def convert_to_3d(self,
                      points: np.ndarray,
                      detection: Detection
                      ) -> BoundingBox3D:

        center_x = detection.box.center.position.x
        center_y = detection.box.center.position.y
        size_x = detection.box.size.x
        size_y = detection.box.size.y

        bb_min_x = int(center_x - size_x / 2.0)
        bb_min_y = int(center_y - size_y / 2.0)
        bb_max_x = int(center_x + size_x / 2.0)
        bb_max_y = int(center_y + size_y / 2.0)

        # masks for limiting the pc using bounding box
        mask_y = np.logical_and(
            bb_min_y <= np.arange(points.shape[0]),
            bb_max_y >= np.arange(points.shape[0])
        )
        mask_x = np.logical_and(
            bb_min_x <= np.arange(points.shape[1]),
            bb_max_x >= np.arange(points.shape[1])
        )

        mask = np.ix_(mask_y, mask_x)

        masked_points = points[mask].reshape(-1, 3)
        masked_points = masked_points[~np.isnan(masked_points).any(axis=1)]

        if masked_points.size == 0:
            return None

        # filter points with clustering
        labels = KMeans(
            init="k-means++",
            n_clusters=2,
            n_init="auto",
            max_iter=300,
            algorithm="lloyd"
        ).fit_predict(masked_points[:, 2].reshape(-1, 1))

        filtered_points = masked_points[labels == 0]
        filtered_points_1 = masked_points[labels == 1]

        if np.min(filtered_points) > np.min(filtered_points_1):
            filtered_points = filtered_points_1

        # max and min
        max_x = np.max(filtered_points[:, 0])
        max_y = np.max(filtered_points[:, 1])
        max_z = np.max(filtered_points[:, 2])

        min_x = np.min(filtered_points[:, 0])
        min_y = np.min(masked_points[:, 1])
        min_z = np.min(filtered_points[:, 2])

        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = float((max_x + min_x) / 2)
        msg.center.position.y = float((max_y + min_y) / 2)
        msg.center.position.z = float((max_z + min_z) / 2)
        msg.size.x = float(max_x - min_x)
        msg.size.y = float(max_y - min_y)
        msg.size.z = float(max_z - min_z)

        return msg

    @staticmethod
    def transform_3d_box(
        box: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray
    ) -> BoundingBox3D:

        # position
        position = BBox3DNode.qv_mult(
            rotation,
            np.array([box.center.position.x,
                      box.center.position.y,
                      box.center.position.z])
        ) + translation

        box.center.position.x = position[0]
        box.center.position.y = position[1]
        box.center.position.z = position[2]

        # size
        size = BBox3DNode.qv_mult(
            rotation,
            np.array([box.size.x,
                      box.size.y,
                      box.size.z])
        )

        box.size.x = abs(size[0])
        box.size.y = abs(size[1])
        box.size.z = abs(size[2])

        return box

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    rclpy.spin(BBox3DNode())
    rclpy.shutdown()
