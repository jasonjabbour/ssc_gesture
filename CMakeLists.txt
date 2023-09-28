cmake_minimum_required(VERSION 2.8.3)
project(ssc_joystick)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra ${CMAKE_CXX_FLAGS}")

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roslint
  roscpp
  nodelet
  std_msgs
  automotive_platform_msgs
  automotive_navigation_msgs
  joy
  rospy
  message_generation
  message_runtime
)

## Add message files
add_message_files(
  FILES
  Gesture.msg
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp nodelet std_msgs automotive_platform_msgs automotive_navigation_msgs joy rospy message_runtime
  LIBRARIES ${PROJECT_NAME}_nodelet
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# nodelet
add_library(${PROJECT_NAME}_nodelet
    src/${PROJECT_NAME}_nodelet.cpp
)

target_link_libraries(${PROJECT_NAME}_nodelet
    ${catkin_LIBRARIES}
)

# node
add_executable(${PROJECT_NAME}_node
    src/ssc_joystick_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
    ${catkin_LIBRARIES}
)

# nodelet for ssc_gesture
add_library(ssc_gesture_nodelet
    src/ssc_gesture_nodelet.cpp
)

target_link_libraries(ssc_gesture_nodelet
    ${catkin_LIBRARIES}
)

# node for ssc_gesture
add_executable(ssc_gesture_node
    src/ssc_gesture_node.cpp
)

target_link_libraries(ssc_gesture_node
    ${catkin_LIBRARIES}
)

roslint_cpp()

#############
## Install ##
#############

install(TARGETS
    ${PROJECT_NAME}_node
    ssc_gesture_node
    ${PROJECT_NAME}_nodelet
    ssc_gesture_nodelet
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES nodelets.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

