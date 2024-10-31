ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO} AS deps

# Create ros2_ws and copy files
WORKDIR /root/ros2_ws
SHELL ["/bin/bash", "-c"]
COPY . /root/ros2_ws/src

# Install dependencies
RUN apt-get update \
    && apt-get -y --quiet --no-install-recommends install \
    gcc \
    git \
    python3 \
    python3-pip
RUN pip3 install -r src/requirements.txt
RUN rosdep install --from-paths src --ignore-src -r -y
RUN pip3 install sphinx==8.0.0 sphinx-rtd-theme==3.0.0

# Colcon the ws
FROM deps AS builder
ARG CMAKE_BUILD_TYPE=Release
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# Source the ROS2 setup file
RUN echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Run a default command, e.g., starting a bash shell
CMD ["bash"]
