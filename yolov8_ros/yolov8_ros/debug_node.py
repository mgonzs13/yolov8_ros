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


import cv2
import random

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose


class DebugNode(Node):

    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)

        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=qos_profile_sensor_data)
        detections_sub = message_filters.Subscriber(
            self, Detection2DArray, "detections", qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 10, 0.1)
        self._synchronizer.registerCallback(self.detections_cb)

    def detections_cb(self, img_msg: Image, detection_msg: Detection2DArray) -> None:

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)

        detection: Detection2D
        for detection in detection_msg.detections:

            # get detection info
            hypothesis: ObjectHypothesisWithPose = detection.results[0]
            label = hypothesis.hypothesis.class_id
            score = hypothesis.hypothesis.score
            box: BoundingBox2D = detection.bbox
            track_id = detection.id

            # draw boxes for debug
            if label not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                box_data = random.randint(0, 255)
                self._class_to_color[label] = (r, g, box_data)
            color = self._class_to_color[label]

            min_pt = (round(box.center.position.x - box.size_x / 2.0),
                      round(box.center.position.y - box.size_y / 2.0))
            max_pt = (round(box.center.position.x + box.size_x / 2.0),
                      round(box.center.position.y + box.size_y / 2.0))
            cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

            label = "{} ({}) ({:.3f})".format(label, str(track_id), score)
            pos = (min_pt[0] + 5, min_pt[1] + 25)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, label, pos, font,
                        1, color, 1, cv2.LINE_AA)

        # publish dbg image
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                           encoding=img_msg.encoding))


def main():
    rclpy.init()
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
