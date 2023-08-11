# defines a YOLOv8 detection result

# class probability
int32 class_id
string class_name
float64 score

# ID for tracking
string id

# 2D bounding box surrounding the object in pixels
yolov8_msgs/BoundingBox2D bbox

# 3D bounding box surrounding the object in meters
yolov8_msgs/BoundingBox3D bbox3d

# segmentation mask of the detected object
# it is only the boundary of the segmented object
yolov8_msgs/Mask mask

# keypoints for human pose estimation
yolov8_msgs/KeyPoint2DArray keypoints

# keypoints for human pose estimation
yolov8_msgs/KeyPoint3DArray keypoints3d