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
