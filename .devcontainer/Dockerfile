# Use the official Ubuntu 18.04 as base image
FROM ubuntu:18.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libxkbcommon-x11-0 \
    xclip \
    xsel

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

# Download and build Python 3.10 from source
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure && \
    make && \
    make install

# Install Python 3 pip and required packages
# For VIP LLaVA you must use this instead of: transformers==4.36.0
RUN apt-get install -y python3-pip && \
    pip3 install --upgrade setuptools pip && \
    pip3 install --upgrade pip && \
    pip3 install scikit-build && \
    pip3 install opencv-python && \ 
    pip3 install mediapipe && \
    pip3 install tensorflow && \
    pip3 install PyYAML && \ 
    pip3 install rospkg && \
    pip3 install imutils && \
    pip3 install ultralytics && \
    pip3 install ftfy && \
    pip3 install regex && \
    pip3 install tqdm && \
    pip3 install git+https://github.com/huggingface/transformers.git && \
    pip3 install bitsandbytes==0.41.3 && \
    pip3 install accelerate==0.25.0 && \
    pip3 install kivy[full]

# Install PyTorch compatible with CUDA 11.5
# MIGHT NEED TO CHANGE CUDA VERSION DEPENDING ON YOUR GPU
# YOU CAN FIND OUT BY RUNNING ON THE HOST MACHINE: nvcc --version
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115

# Install CLIP from the GitHub repository
RUN pip3 install git+https://github.com/openai/CLIP.git

RUN apt-get install -y \
    x11-apps

# Source ROS setup.bash in the bashrc file for the user
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# Initialize rosdep and update it
RUN rosdep init && \
    rosdep update

# Expose a shell when the container is run
CMD ["/bin/bash"]









# # # Use the official Ubuntu 18.04 as base image
# # FROM ubuntu:18.04

# # # Set environment variables to avoid interactive prompts during installation
# # ENV DEBIAN_FRONTEND=noninteractive

# # # Update the package list and install necessary tools
# # RUN apt-get update && apt-get install -y \
# #     curl \
# #     gnupg \
# #     software-properties-common

# # # Add deadsnakes PPA for newer versions of Python
# # RUN add-apt-repository ppa:deadsnakes/ppa

# # # Update package list again
# # RUN apt-get update

# # # Install Python 3.10 without the -dev package
# # RUN apt-get install -y python3.10

# # # # Set Python 3.10 as the default Python version
# # # RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
# # #     update-alternatives --set python3 /usr/bin/python3.10

# # # # Ensure our version of Python is used
# # # RUN ln -s /usr/bin/python3 /usr/bin/python

# # # Add ROS repository key
# # RUN curl -sSL http://packages.ros.org/ros.key | apt-key add -

# # # Add ROS repository to the sources list
# # RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros.list

# # # Update the package list again to include the ROS repository
# # RUN apt-get update

# # # Install ROS Melodic packages
# # # You can customize this list based on your specific requirements
# # RUN apt-get install -y \
# #     ros-melodic-desktop-full

# # # Source ROS setup.bash in the bashrc file so ROS environment variables are automatically set
# # RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# # # Install additional dependencies and tools
# # RUN apt-get install -y \
# #     python-rosdep \
# #     python-rosinstall \
# #     python-rosinstall-generator \
# #     python-wstool \
# #     build-essential \
# #     ros-melodic-nodelet ros-melodic-roslint ros-melodic-automotive-platform-msgs \
# #     ros-melodic-automotive-navigation-msgs ros-melodic-joy

# # # Install Python 3 pip and required packages
# # RUN apt-get install -y python3-pip && \
# #     pip3 install --upgrade setuptools pip && \
# #     pip3 install --upgrade pip && \
# #     pip3 install scikit-build && \
# #     pip3 install opencv-python==4.5.3.56 && \ 
# #     pip3 install mediapipe && \
# #     pip3 install tensorflow

# # # Source ROS setup.bash in the bashrc file for the user
# # RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# # # Initialize rosdep and update it
# # RUN rosdep init && \
# #     rosdep update

# # # Expose a shell when the container is run
# # CMD ["/bin/bash"]


# # ----------------------------
# # # Use the official Ubuntu 18.04 as base image
# FROM ubuntu:18.04

# # Set environment variables to avoid interactive prompts during installation
# ENV DEBIAN_FRONTEND=noninteractive

# # Update the package list and install necessary tools
# RUN apt-get update && apt-get install -y \
#     curl \
#     gnupg \
#     software-properties-common

# # # Add deadsnakes PPA for newer versions of Python
# # RUN add-apt-repository ppa:deadsnakes/ppa

# # # Update package list again
# # RUN apt-get update

# # # Install Python 3.10 without the -dev package
# # RUN apt-get install -y python3.10

# # Add ROS repository key
# RUN curl -sSL http://packages.ros.org/ros.key | apt-key add -

# # Add ROS repository to the sources list
# RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros.list

# # Update the package list again to include the ROS repository
# RUN apt-get update

# # Install ROS Melodic packages
# # You can customize this list based on your specific requirements
# # RUN apt-get install -y \
# #     ros-melodic-desktop-full

# # # Source ROS setup.bash in the bashrc file so ROS environment variables are automatically set
# # RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# # # Install additional dependencies and tools
# # RUN apt-get install -y \
# #     python-rosdep \
# #     python-rosinstall \
# #     python-rosinstall-generator \
# #     python-wstool \
# #     build-essential \
# #     ros-melodic-nodelet ros-melodic-roslint ros-melodic-automotive-platform-msgs \
# #     ros-melodic-automotive-navigation-msgs ros-melodic-joy

# # # Install Python 3 pip and required packages
# # RUN apt-get install -y python3-pip && \
# #     pip3 install --upgrade setuptools pip && \
# #     pip3 install --upgrade pip && \
# #     pip3 install scikit-build && \
# #     pip3 install opencv-python==4.5.3.56 && \ 
# #     pip3 install mediapipe && \
# #     pip3 install tensorflow

# # # Source ROS setup.bash in the bashrc file for the user
# # RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc

# # # Initialize rosdep and update it
# # RUN rosdep init && \
# #     rosdep update

# # Expose a shell when the container is run
# CMD ["/bin/bash"]


# # # Upgrade pip and install required packages
# # RUN python3.10 -m pip install --upgrade pip && \
# #     python3.10 -m pip install scikit-build && \
# #     python3.10 -m pip install opencv-python==4.5.3.56 && \
# #     python3.10 -m pip install mediapipe && \
# #     python3.10 -m pip install tensorflow