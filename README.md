# yolov8_ros

ROS 2 wrap for [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) to perform object detection and tracking.

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

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

## Instance Segmentation

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py model:=yolov8m-seg.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

## Human Pose

```shell
$ ros2 launch yolov8_bringup yolov8.launch.py model:=yolov8m-pose.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

## 3D Object Detection

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1rkQsu3-JmfEvJVGqIHS3uXEBdCfDzDQ1)](https://drive.google.com/file/d/1rkQsu3-JmfEvJVGqIHS3uXEBdCfDzDQ1/view?usp=sharing)

## 3D Object Detection (Using Instance Segmentation Masks)

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py model:=yolov8m-seg.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=16NvYp5ABrraHkdKzWeytsidGUK4vqyFs)](https://drive.google.com/file/d/16NvYp5ABrraHkdKzWeytsidGUK4vqyFs/view?usp=sharing)

## 3D Human Pose

```shell
$ ros2 launch yolov8_bringup yolov8_3d.launch.py model:=yolov8m-pose.pt
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1lO3qSud6cSuaqHb-gB0tpq9Nu5hZt72Z)](https://drive.google.com/file/d/1lO3qSud6cSuaqHb-gB0tpq9Nu5hZt72Z/view?usp=sharing)
