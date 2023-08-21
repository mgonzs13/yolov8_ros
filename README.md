# yolov8_ros

ROS 2 wrap for [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation and human pose estamation. There are also 3D versions of object detection and human pose estimation based on Point Cloud.

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone https://github.com/mgonzs13/yolov8_ros.git
$ pip3 install -r yolov8_ros/requirements.txt
$ cd ~/ros2_ws
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build
```

## Usage

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py
```

## Demos

## Object Detection

This is the standard behavior of YOLOv8, which includes object tracking.

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

## Instance Segmentation

Instance masks are the borders of the detected objects, not the all the pixels inside the masks.

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py model:=yolov8m-seg.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

## Human Pose

Online persons are detected along with their keypoints.

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py model:=yolov8m-pose.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

## 3D Object Detection

The 3D bounding boxes are calculated filtering the Point Cloud data from an RGB-D camera using the 2D bounding box. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## 3D Object Detection (Using Instance Segmentation Masks)

In this, the Point Cloud data is filtered using the max and min values obtained from the instance masks. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py model:=yolov8m-seg.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

## 3D Human Pose

Each keypoint is projected in the Point Cloud and visualized using purple spheres. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py model:=yolov8m-pose.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)
