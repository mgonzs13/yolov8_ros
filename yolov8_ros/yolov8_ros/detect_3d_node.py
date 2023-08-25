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
from yolov8_msgs.msg import KeyPoint3D
from yolov8_msgs.msg import KeyPoint3DArray


class Detect3DNode(Node):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.target_frame = self.get_parameter(
            "target_frame").get_parameter_value().string_value
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.maximum_detection_threshold = self.get_parameter(
            "maximum_detection_threshold").get_parameter_value().double_value

        # aux
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

        # subs
        self.points_sub = message_filters.Subscriber(
            self, PointCloud2, "points",
            qos_profile=qos_profile_sensor_data)
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.points_sub, self.detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.on_detections)

    def on_detections(self,
                      points_msg: PointCloud2,
                      detections_msg: DetectionArray,
                      ) -> None:

        # check if there are detections
        if not detections_msg.detections:
            return

        transform = self.get_transform(points_msg.header.frame_id)

        if transform is None:
            return

        points = np.frombuffer(points_msg.data, np.float32).reshape(
            points_msg.height, points_msg.width, -1)[:, :, :3]

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header

        for detection in detections_msg.detections:
            bbox3d = self.convert_bb_to_3d(points, detection)

            if bbox3d is not None:
                new_detections_msg.detections.append(detection)

                bbox3d = Detect3DNode.transform_3d_box(
                    bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                new_detections_msg.detections[-1].bbox3d = bbox3d

                keypoints3d = self.convert_keypoints_to_3d(
                    points, detection)
                keypoints3d = Detect3DNode.transform_3d_keypoints(
                    keypoints3d, transform[0], transform[1])
                keypoints3d.frame_id = self.target_frame
                new_detections_msg.detections[-1].keypoints3d = keypoints3d

        self._pub.publish(new_detections_msg)

    def convert_bb_to_3d(self,
                         points: np.ndarray,
                         detection: Detection
                         ) -> BoundingBox3D:

        if detection.mask.data:
            detection_mask_x = np.array(
                [point.x for point in detection.mask.data])
            detection_mask_y = np.array(
                [point.y for point in detection.mask.data])

            bb_min_x = int(np.min(detection_mask_x))
            bb_min_y = int(np.min(detection_mask_y))
            bb_max_x = int(np.max(detection_mask_x))
            bb_max_y = int(np.max(detection_mask_y))

            center_x = (bb_min_x + bb_max_x) / 2
            center_y = (bb_min_y + bb_max_y) / 2

        else:
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size.x
            size_y = detection.bbox.size.y

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

        # maximum_detection_threshold
        center_point = points[int(center_y)][int(center_x)]
        z_diff = np.abs(masked_points[:, 2] - center_point[2])
        mask_z = z_diff <= self.maximum_detection_threshold
        masked_points = masked_points[mask_z]

        # remove nan
        filtered_points = masked_points[~np.isnan(masked_points).any(axis=1)]

        if masked_points.shape[0] < 2:
            return None

        # max and min
        max_x = np.max(filtered_points[:, 0])
        max_y = np.max(filtered_points[:, 1])
        max_z = np.max(filtered_points[:, 2])

        min_x = np.min(filtered_points[:, 0])
        min_y = np.min(filtered_points[:, 1])
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

    def convert_keypoints_to_3d(self,
                                points: np.ndarray,
                                detection: Detection
                                ) -> KeyPoint3DArray:

        msg_array = KeyPoint3DArray()

        for p in detection.keypoints.data:

            if int(p.point.y) >= points.shape[0] or int(p.point.x) >= points.shape[1]:
                continue

            p3d = points[int(p.point.y)][int(p.point.x)]

            if not np.isnan(p3d).any():
                msg = KeyPoint3D()
                msg.point.x = float(p3d[0])
                msg.point.y = float(p3d[1])
                msg.point.z = float(p3d[2])
                msg.id = p.id
                msg.score = p.score
                msg_array.data.append(msg)

        return msg_array

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from point cloud frame to target_frame
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

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray
    ) -> BoundingBox3D:

        # position
        position = Detect3DNode.qv_mult(
            rotation,
            np.array([bbox.center.position.x,
                      bbox.center.position.y,
                      bbox.center.position.z])
        ) + translation

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # size
        size = Detect3DNode.qv_mult(
            rotation,
            np.array([bbox.size.x,
                      bbox.size.y,
                      bbox.size.z])
        )

        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = Detect3DNode.qv_mult(
                rotation,
                np.array([
                    point.point.x,
                    point.point.y,
                    point.point.z
                ])
            ) + translation

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

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
    rclpy.spin(Detect3DNode())
    rclpy.shutdown()
