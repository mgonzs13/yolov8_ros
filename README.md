# yolov8_ros

ROS 2 wrap for [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation and human pose estimation. There are also 3D versions of object detection and human pose estimation based on depth images.

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

### YOLOv8

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py
```

#### Parameters

- **model**: YOLOv8 model (default: yolov8m.pt)
- **tracker**: tracker file (default: bytetrack.yaml)
- **device**: GPU/CUDA (default: cuda:0)
- **enable**: wether to start YOLOv8 enabled (default: True)
- **threshold**: detection threshold (default: 0.5)
- **input_image_topic**: camera topic of RGB images (default: /camera/rgb/image_raw)

### YOLOv8 3D

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py
```

#### Parameters

- **model**: YOLOv8 model (default: yolov8m.pt)
- **tracker**: tracker file (default: bytetrack.yaml)
- **device**: GPU/CUDA (default: cuda:0)
- **enable**: wether to start YOLOv8 enabled (default: True)
- **threshold**: detection threshold (default: 0.5)
- **input_image_topic**: camera topic of RGB images (default: /camera/rgb/image_raw)
- **input_depth_topic**: camera topic of depth images (default: /camera/depth/image_raw)
- **input_depth_info_topic**: camera topic for info data (default: /camera/depth/camera_info)
- **depth_image_units_divisor**: divisor to convert the depth image into metres (default: 1000)
- **target_frame**: frame to transform the 3D boxes (default: base_link)
- **maximum_detection_threshold**: maximum detection threshold in the z axis (default: 0.3)

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

The 3D bounding boxes are calculated filtering the depth image data from an RGB-D camera using the 2D bounding box. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## 3D Object Detection (Using Instance Segmentation Masks)

In this, the depth image data is filtered using the max and min values obtained from the instance masks. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py model:=yolov8m-seg.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

## 3D Human Pose

Each keypoint is projected in the depth image and visualized using purple spheres. Only objects with a 3D bounding box are visualized in the 2D image.

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py model:=yolov8m-pose.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)
