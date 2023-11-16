# ROS SSC Joystick Application #

[![CircleCI](https://circleci.com/gh/astuff/ssc_joystick/tree/master.svg?style=svg)](https://circleci.com/gh/astuff/ssc_joystick/tree/master)

## Overview

The ssc_joystick ROS package is used to verify that the SSC is operational. 
It is similar to the ROS joystick node with some notable exceptions. 

This application is intended to convert the user's joystick commands into gear, steering, speed, and turn signal commands
to pass to the AutonomouStuff Speed and Steering Control software.  
Use the launch file provided to start this module and the joy node.  
The SSC and drive-by-wire modules need to be started outside of this launch file.

## Installation

The `ssc_joystick` package can be installed using our AutonomouStuff apt repo:

```sh
sudo apt install apt-transport-https
sudo sh -c 'echo "deb [trusted=yes] https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
sudo apt update
sudo apt install ros-$ROS_DISTRO-ssc-joystick
```

## Control Scheme

The `ssc_joystick` package is intended to be used with the Logitech F310 gamepad controller as described below.
However, if you wish to use a different controller, the ROS parameters specified in the `config/params.yaml` file can be modified to enable any other controller or control scheme.

![Left: Front Layout of logitech Controller; Right: Side-button layout of logitech controller
](/controller_img.png "controller_img.png")

| Action | Button | Notes |
| - | - | - |
| **Enable/Disable** | **Center region** | |
| Enable | BACK and START | Buttons must be pressed simultaneously to engage |
| Disable | BACK | button must be pressed to disengage |
| **Gear Selection** | **Button Pad (right-hand side)** | |
| Drive | A | |
| Reverse | B | |
| Neutral | X | |
| Park | Y | |
| **Speed and Steering** | | |
| Speed setpoint adjust | Directional Pad Up/Down | Adjust speed in steps |
| Steering setpoint adjust | Directional Pad Left/Right | Adjust steering in steps |
| Brake Override | Left Trigger | Applies brakes immediately |
| Steering Override | Right Stick | Steers immediately |
| **Other** | | |
| Left turn signal | Left Bumper | Activate the left turn signal |
| Right turn signal | Right Bumper | Activate the right turn signal |

## Functional Overview

Once the software is running, push either both engage buttons on the joystick or both the cruise control set/dec and
decrease gap buttons on the steering wheel (not supported on all platforms) to take control with the joystick,
ENGAGED will be output to the screen. The desired speed defaults to 0 mph when you engage, so the software will
automatically engage the brakes.
You can also choose to enable the speed and steering modules individually (not supported on all platforms).

You can then press the drive button to place the gear in drive, since the desired speed is still zero the brakes
will still be applied.  Pressing the speed up and down buttons will increment and decrement by the configured step
amount, limiting the speed between 0 and the maximum speed set.  Any time a speed button is pressed the new desired
speed will be output to the screen.  To bring the vehicle to a stop, step the speed back down to zero, and the control
software will gently apply the brakes to bring the vehicle to a stop.  Pressing the brake trigger will cause the
desired velocity to drop to zero, effectively applying the brakes. (The brake pedal can always be applied by the driver
to override the joystick control mode.)

The steering gain and exponent convert the steering joystick to a desired curvature which is passed down to the
steering model.  The gain defines the maximum curvature, so the default of 0.12 1/meters allows for a minimum turning
radius of about 8 meters.  The exponent controls the shape of the response: a number closer to 2 or above will mean
small joystick movements will translate to very small desired curvatures and therefore steering wheel angles,
a number closer to 1 will mean the curvature varies more linearly across the full joystick range.

The curvature command can also be changed by pressing the left and right steering buttons.  The updated curvature
will be output to the screen.  The steering joystick will override the value set with the buttons.

The left and right turn signals can also be controlled with the buttons.  The turn signals will stay on as long
as the button is pressed.

Pressing the disengage button on the joystick will give control back to the driver.
On supported vehicles, pressing both the cruise control set/inc and increase gap buttons on the steering wheel will
result in a disengage.
On vehicles with the default override behavior, any drive override on the brakes, throttle, or steering wheel will also
result in returning control to the driver.
DISENGAGE or a message with information for an override will be sent to the screen.

It is also intended that this application be used as an example of how to interface to the speed and steering control
software modules and can be used as a starting point for the development of higher level autonomy features.

# ssc_gesture

### Setup Commands

```sh

# Outside of docker to allow docker to access host's X server
xhost local:root

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

