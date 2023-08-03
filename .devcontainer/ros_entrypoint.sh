#!/bin/bash
set -e

# source ros package
source "/opt/ros/$ROS_DISTRO/setup.bash"

# execute command passed into docker run
exec "$@"