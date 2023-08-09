# Defines a YOLOv8 detection result

# class probability
vision_msgs/ObjectHypothesis hypothesis

# ID for tracking
string id

# 3D bounding box surrounding the object
# can be used as a 2D bouding box
vision_msgs/BoundingBox3D box

# segmentation mask of the detected object
# it is only the boundary of the segmented object
yolov8_msgs/Mask mask

# keypoint array for human pose
yolov8_msgs/KeyPointArray keypoints