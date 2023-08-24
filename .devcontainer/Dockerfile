# Use the official Ubuntu 18.04 as base image
FROM ubuntu:18.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary tools
RUN apt-get update && apt-get install -y \
    curl \
    gnupg

# Add ROS repository key
RUN curl -sSL http://packages.ros.org/ros.key | apt-key add -

# Add ROS repository to the sources list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros.list

# Update the package list again to include the ROS repository
RUN apt-get update

# Install ROS Melodic packages
# You can customize this list based on your specific requirements
RUN apt-get install -y \
    ros-melodic-desktop-full

# Source ROS setup.bash in the bashrc file so ROS environment variables are automatically set
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# Install additional dependencies and tools
RUN apt-get install -y \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential \
    ros-melodic-nodelet ros-melodic-roslint ros-melodic-automotive-platform-msgs \
    ros-melodic-automotive-navigation-msgs ros-melodic-joy

# Install Python 3 pip and required packages
RUN apt-get install -y python3-pip && \
    pip3 install --upgrade setuptools pip && \
    pip3 install --upgrade pip && \
    pip3 install scikit-build && \
    pip3 install opencv-python==4.5.3.56 && \ 
    pip3 install mediapipe && \
    pip3 install tensorflow

# Source ROS setup.bash in the bashrc file for the user
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# Initialize rosdep and update it
RUN rosdep init && \
    rosdep update

# Expose a shell when the container is run
CMD ["/bin/bash"]


# # Install Python 3.10 and pip
# RUN apt-get install -y python3.10 python3.10-distutils && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python3.10 get-pip.py && \
#     rm get-pip.py

# RUN apt-get -y update && apt-get install software-properties-common \
# && add-apt-repository ppa:deadsnakes/ppa && apt install python3.10

# # Install Python 3.10 and pip
# RUN add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.10 python3.10-distutils && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python3.10 get-pip.py && \
#     rm get-pip.py

# # Upgrade pip and install required packages
# RUN python3.10 -m pip install --upgrade pip && \
#     python3.10 -m pip install scikit-build && \
#     python3.10 -m pip install opencv-python==4.5.3.56 && \
#     python3.10 -m pip install mediapipe && \
#     python3.10 -m pip install tensorflow