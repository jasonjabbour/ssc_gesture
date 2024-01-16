# ssc_gesture

### Outside Docker Setup Commands
```sh

# Outside of docker to allow docker to access host's X server
xhost local:root

# Create workspace
mkdir /photondrive_ws/src -p

# Clone Github Repo
cd /photondrive_ws/src
git clone https://github.com/jasonjabbour/ssc_gesture.git

# Create directory to store LLaVA model
cd /photondrive_ws/src
mkdir llava_model
```

### Inside Docker Setup Commands

```sh
# Make the Workspace
cd /tmp/photondrive_ws
catkin_make

# Source the overlay
source devel/setup.bash

# Install package dependencies
cd /tmp/photondrive_ws/src/ssc_gesture
pip3 install -e .

```

### Launch Scripts

```bash

cd /tmp/photondrive_ws

# To Launch Gesture Recognition Alone:
roslaunch ssc_joystick gesture_detection.launch

# To Launch Joystick Alone:
roslaunch ssc_joystick ssc_joystick.launch

# Launch Gesture and SSC Controller
roslaunch ssc_joystick ssc_gesture.launch

```

### Complementary Systems

```bash
# In the first terminal: 
roslaunch pacmod pacmod.launch​

# In the second terminal: 
roslaunch ssc_pm_gem_e4 speed_steering_control.launch​
```

### Troubleshooting

If you don't see any errors on any of the terminal screens, you should be able to enable using the controller. the controls for using it are available in the README here (NOTE: Make sure you have the E-stop release before you test). Let me know if you have any issues with this and I can help you out. 

Fix Namespace:
Within the opt/ros/melodic/share/pacmod/launch file add ns="pacmod" to the node pkg="socketcan_bridge" and pkg="pacmod" elements. 

### Info

This repository estimates full body poses using the Python version of MediaPipe. It uses the pose detection model from Mediapipe to identify the 32 landmarks on a full body. The keypoint model is responsible for classifying static poses while the point history and multipoint history classifies left hand motion and full body motion, respectively. 

This code has been adapted from https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/app.py. 

The current classifications for full body motion (as seen in multipoint history) are default, turn left, turn right, move forward, and slow down. The multipoint history model only tracks motion data from the first 25 landmarks which include all body landmarks from Mediapipe’s pose detection model except for the leg landmarks. The point history model tracks motion data for the left wrist, but it can be easily adapted to accomodate any  landmark from the 32 full body landmarks.

Push one of the following lower-case letters after running app.py to alter the csv files:
a -> This will prompt the user for a new multipoint classifier label that will be added to multi_point_history_classifier_label.csv
The following 4 modes are used for logging new data:
n -> (mode 0) This key returns the model to mode 0 where it cannot log any new data.
k -> (mode 1) While in mode 1 and pushing a number between 0 and 9 (inclusive), stationary image data will be recorded for keypoints.
h -> (mode 2) While in mode 2 and pushing a number between 0 and 9 (inclusive), data will be recorded for left wrist motion.
b -> (mode 3) While in mode 3 and pushing a number between 0 and 9 (inclusive), data will be recorded for full motion.

