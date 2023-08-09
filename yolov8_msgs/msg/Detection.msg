# Defines a YOLOv8 detection result

# class probability
vision_msgs/ObjectHypothesis hypothesis

# 3D bounding box surrounding the object
# can be used as a 2D bouding box
vision_msgs/BoundingBox3D box

# segmentation mask of the detected object
yolov8_msgs/Mask mask

# ID used for tracking
string id
