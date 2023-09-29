/*
 * AutonomouStuff, LLC ("COMPANY") CONFIDENTIAL
 * Unpublished Copyright (c) 2009-2021 AutonomouStuff, LLC, All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains the property of COMPANY. The intellectual and technical
 * concepts contained herein are proprietary to COMPANY and may be covered by U.S. and Foreign Patents, patents in
 * process, and are protected by trade secret or copyright law. Dissemination of this information or reproduction of
 * this material is strictly forbidden unless prior written permission is obtained from COMPANY.  Access to the source
 * code contained herein is hereby forbidden to anyone except current COMPANY employees, managers or contractors who
 * have executed Confidentiality and Non-disclosure agreements explicitly covering such access.
 *
 * The copyright notice above does not evidence any actual or intended publication or disclosure  of  this source code,
 * which includes information that is confidential and/or proprietary, and is a trade secret, of  COMPANY.   ANY
 * REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS  SOURCE
 * CODE  WITHOUT  THE EXPRESS WRITTEN CONSENT OF COMPANY IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
 * INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR
 * IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR SELL ANYTHING THAT IT
 * MAY DESCRIBE, IN WHOLE OR IN PART.
 */

#include "ssc_joystick/ssc_gesture_nodelet.hpp"

#include <algorithm>
#include <math.h>
#include <string>
#include <utility>

#include <pluginlib/class_list_macros.h>

namespace astuff
{
namespace
{
template <typename T>
T clamp(const T value, T bound1, T bound2)
{
  if (bound1 > bound2)
  {
    std::swap(bound1, bound2);
  }

  if (value < bound1)
  {
    return bound1;
  }
  else if (value > bound2)
  {
    return bound2;
  }
  return value;
}
}  // namespace

void SscGestureNl::onInit()
{
  nh_ = getNodeHandle();
  pnh_ = getPrivateNodeHandle();
  loadParams();

  if (engage_speed_module_ || engage_steering_module_)
  {
    NODELET_INFO("SPEED MODULE MODE: %d STEERING MODULE MODE: %d", engage_speed_module_, engage_steering_module_);

    if (engage_speed_module_ && engage_steering_module_)
    {
      NODELET_INFO("SPEED AND STEERING CONTROL SET TO ENGAGE");
    }
  }
  else
  {
    NODELET_WARN("NO MODULE HAS BEEN SET TO ENGAGE, SSC WILL NOT BE ACTIVE");
  }

  // Giva joystick a few seconds to start up
  last_joystick_msg_timestamp_ = ros::Time::now().toSec() + 5.0;

  // Subscribers
  joy_sub_ = nh_.subscribe("joy", 10, &SscGestureNl::joystickCallback, this);
  joy_fault_sub_ = nh_.subscribe("diagnostics", 10, &SscGestureNl::diagnosticCallback, this);
  gear_sub_ = nh_.subscribe("gear_feedback", 10, &SscGestureNl::gearFeedbackCallback, this);
  velocity_sub_ = nh_.subscribe("velocity_accel_cov", 10, &SscGestureNl::velocityCallback, this);
  adas_input_sub_ = nh_.subscribe("adas_input", 10, &SscGestureNl::inputAdasCallback, this);
  module_state_sub_ = nh_.subscribe("module_states", 10, &SscGestureNl::moduleStateCallback, this);

  if (training_wheels_mode){
    gesture_topic_sub_ = nh_.subscribe("training_wheels_gesture_topic", 10, &SscGestureNl::gestureClassCallback, this);
  }
  else {
    gesture_topic_sub_ = nh_.subscribe("gesture_topic", 10, &SscGestureNl::gestureClassCallback, this);
  }
  
  // Publishers
  gear_cmd_pub_ = nh_.advertise<automotive_platform_msgs::GearCommand>("gear_select", 1);
  turn_signal_cmd_pub_ = nh_.advertise<automotive_platform_msgs::TurnSignalCommand>("turn_signal_command", 1);
  speed_cmd_pub_ = nh_.advertise<automotive_platform_msgs::SpeedMode>("arbitrated_speed_commands", 1);
  steer_cmd_pub_ = nh_.advertise<automotive_platform_msgs::SteerMode>("arbitrated_steering_commands", 1);

  // Vehicle command timer
  vehicle_cmd_timer_ = nh_.createTimer(ros::Duration(publish_interval_), &SscGestureNl::publishVehicleCommand, this);
  NODELET_INFO("ssc_joystick initialized");
}

void SscGestureNl::loadParams()
{
  pnh_.param("publish_interval", publish_interval_, 0.05f);
  pnh_.param("joystick_fault_timeout", joystick_fault_timeout_, 3.0f);

  pnh_.param<std::string>("veh_controller_name", veh_controller_name_, "/ssc/veh_controller");
  pnh_.param("engage_speed_module", engage_speed_module_, true);
  pnh_.param("engage_steering_module", engage_steering_module_, true);
  pnh_.param("engage1_button", engage1_button_, 6);
  pnh_.param("engage2_button", engage2_button_, 7);

  pnh_.param("park_button", park_button_, 3);
  pnh_.param("neutral_button", neutral_button_, 2);
  pnh_.param("drive_button", drive_button_, 0);
  pnh_.param("reverse_button", reverse_button_, 1);
  pnh_.param("right_turn_button", right_turn_button_, 5);
  pnh_.param("left_turn_button", left_turn_button_, 4);

  pnh_.param("speed_axes", speed_axes_, 7);
  pnh_.param("speed_up_sign", speed_up_sign_, 1.0f);
  pnh_.param("speed_step", speed_step_, 1.0f);
  pnh_.param("max_speed", max_speed_, 15.0f);
  pnh_.param("acceleration_limit", acceleration_limit_, 2.0f);
  pnh_.param("deceleration_limit", deceleration_limit_, 2.5f);
  pnh_.param("max_deceleration_limit", max_deceleration_limit_, 4.0f);
  deceleration_ = deceleration_limit_;
  pnh_.param("brake_axes", brake_axes_, 2);
  pnh_.param("brake_sign", brake_sign_, 1.0f);

  pnh_.param("steer_btn_axes", steer_btn_axes_, 6);
  pnh_.param("steer_btn_sign", steer_btn_sign_, 1);
  pnh_.param("steer_btn_step", steer_btn_step_, 0.01f);
  pnh_.param("steering_axes", steering_axes_, 3);
  pnh_.param("steering_sign", steering_sign_, 1.0f);
  pnh_.param("steering_exponent", steering_exponent_, 2.5f);
  pnh_.param("max_curvature", max_curvature_, 0.12f);
  pnh_.param("max_curvature_rate", max_curvature_rate_, 0.10f);

  pnh_.param("test_quick_brake", test_quick_brake_, false);
  pnh_.param("quick_brake_speed", quick_brake_speed_, 0.0f);

  NODELET_INFO("Parameters Loaded");
}

void SscGestureNl::joystickCallback(const sensor_msgs::Joy::ConstPtr& msg)
{

  createEngageCommand(msg);

  if (testing_mode_){
    engaged_ = true; 
  }

  if (engaged_)
  {
    createShiftCommand(msg);
    // createSpeedCommand(msg);
    // createSteeringCommand(msg);
    createAuxCommand(msg);
  }
  else
  {
    desired_velocity_ = 0.0;
    desired_curvature_ = 0.0;
  }
}

// NEW
void SscGestureNl::gestureClassCallback(const ssc_joystick::Gesture::ConstPtr& msg)
{

  if (testing_mode_){
    engaged_ = true; 
  }

  if (engaged_)
  {
    createSpeedandSteeringGesture(msg);
    // float x = 0;
  }
  else
  {
    desired_velocity_ = 0.0;
    desired_curvature_ = 0.0;
  }

}


void SscGestureNl::createEngageCommand(const sensor_msgs::Joy::ConstPtr& msg)
{
  if ((msg->buttons.at((uint32_t)engage1_button_) > 0) && (msg->buttons.at((uint32_t)engage2_button_) > 0))
  {

    if (!engage_pressed_)
    {

      if (engaged_)
      {
        disengage();
      }
      else
      {
        
        if (testing_mode_){
          tryToUnsafelyEngage();
        }
        else
        {
          tryToEngage();
        }

      }
      engage_pressed_ = true;
    }
  }
  else if ((msg->buttons.at((uint32_t)engage1_button_) > 0) || (msg->buttons.at((uint32_t)engage2_button_) > 0))
  {
    if (engaged_ && !engage_pressed_)
    {
      disengage();
      engage_pressed_ = true;
    }
  }
  else
  {
    engage_pressed_ = false;
  }
}

void SscGestureNl::createShiftCommand(const sensor_msgs::Joy::ConstPtr& msg)
{
  if (msg->buttons.at((uint32_t)park_button_) > 0)
  {
    if (current_velocity_ > 0.1)
    {
      NODELET_WARN("Must be stopped to change to park");
    }
    else
    {
      desired_gear_ = automotive_platform_msgs::Gear::PARK;
      NODELET_INFO("Vehicle Gear in: Park");
    }
  }
  else if (msg->buttons.at((uint32_t)neutral_button_) > 0)
  {
    desired_gear_ = automotive_platform_msgs::Gear::NEUTRAL;
    NODELET_INFO("Vehicle Gear in: Neutral");
  }
  else if (msg->buttons.at((uint32_t)drive_button_) > 0)
  {
    desired_gear_ = automotive_platform_msgs::Gear::DRIVE;
    NODELET_INFO("Vehicle Gear in: Drive");
  }
  else if (msg->buttons.at((uint32_t)reverse_button_) > 0)
  {
    desired_gear_ = automotive_platform_msgs::Gear::REVERSE;
    NODELET_INFO("Vehicle Gear in: Reverse");
  }
}

void SscGestureNl::createSpeedCommand(const sensor_msgs::Joy::ConstPtr& msg)
{
  float speed = msg->axes.at((uint32_t)speed_axes_);
  bool speed_updated = false;
  if (speed > 0.1)
  {
    if (speed_last_ != 1)
    {
      if (!test_quick_brake_ || (test_quick_brake_ && (desired_velocity_ < quick_brake_speed_)))
      {
        desired_velocity_ += speed_up_sign_ * speed_step_;
        speed_updated = true;
      }
      else if (test_quick_brake_ && (desired_velocity_ > quick_brake_speed_))
      {
        desired_velocity_ = 0.0;
        deceleration_ = 0.0;
        speed_updated = true;

        NODELET_INFO("Quick Brake Test: Make sure related SSC values are non-zero.");
      }
    }
    speed_last_ = 1;
  }
  else if (speed < -0.1)
  {
    if (speed_last_ != -1)
    {
      desired_velocity_ -= speed_up_sign_ * speed_step_;
      speed_updated = true;
    }
    speed_last_ = -1;
  }
  else
  {
    speed_last_ = 0;
  }

  float brake = msg->axes.at((uint32_t)brake_axes_);
  if (brake != 0.0 || brake_initialized_)
  {
    brake_initialized_ = true;
    brake *= brake_sign_;
    if (brake < 0.95f)
    {
      if (!brake_active_)
      {
        brake_active_ = true;
        desired_velocity_ = 0.0f;
        speed_updated = true;
      }
      auto map2pt = [](float in, float min_in, float max_in, float min_out, float max_out) {  // NOLINT
        float out;
        if (in <= min_in)
          out = min_out;
        else if (in >= max_in)
          out = max_out;
        else
          out = (in - min_in) / (max_in - min_in) * (max_out - min_out) + min_out;
        return out;
      };
      deceleration_ = map2pt(brake, -0.95f, 0.95f, max_deceleration_limit_, deceleration_limit_);
    }
    else
    {
      if (brake_active_)
      {
        brake_active_ = false;
        // convert from m/s to mph
        desired_velocity_ = current_velocity_ / 0.44704f;
        desired_velocity_ = static_cast<float>(speed_step_ * std::floor(desired_velocity_ / speed_step_));
        speed_updated = true;
        deceleration_ = deceleration_limit_;
      }
    }
  }

  if (speed_updated)
  {
    desired_velocity_ = static_cast<float>(speed_step_ * std::round(desired_velocity_ / speed_step_));

    if (desired_velocity_ > max_speed_)
    {
      desired_velocity_ = max_speed_;
    }
    else if (desired_velocity_ < 0.1)
    {
      desired_velocity_ = 0.0;
    }

    NODELET_INFO("Desired velocity: %f", desired_velocity_);
  }
}

void SscGestureNl::createSteeringCommand(const sensor_msgs::Joy::ConstPtr& msg)
{
  // Extract steering value from the joystick message based on the specified steering axes
  float steering = msg->axes.at((uint32_t)steering_axes_);

  // Check if the absolute value of the steering is significant enough to be considered
  if ((steering > 0.01) || (steering < -0.01))
  {
    // Adjust the raw steering value using the steering sign
    float raw = steering * steering_sign_;

    // Calculate desired curvature based on the transformed steering value
    // The value is raised to the power of steering_exponent_, which allows for exponential steering responses
    desired_curvature_ = std::copysign(std::pow(std::fabs(raw), steering_exponent_) * max_curvature_, raw);

    // Mark that steering is currently active
    steering_active_ = true;
  }
  // If the steering is neutral but was previously active
  else if (steering_active_)
  {
    // Reset desired curvature to zero
    desired_curvature_ = 0.0;

    // Mark that steering is no longer active
    steering_active_ = false;
  }
  // Handle the case where steering is not significant and it was not previously active
  else
  {
    // Extract steering value from joystick message based on the specified button axes for steering
    float steer = msg->axes.at((uint32_t)steer_btn_axes_);

    // Initialize a flag to track if steering has been modified
    bool steer_updated = false;

    // Check if the steer value indicates a positive (rightward) turn
    if (steer > 0.1)
    {
      // Ensure the previous steering command wasn't already turning right
      if (steer_last_ != 1)
      {
        // Increase desired curvature by a fixed step
        desired_curvature_ += steer_btn_sign_ * steer_btn_step_;

        // Mark that steering has been updated
        steer_updated = true;
      }
      // Record the last steering direction as rightward
      steer_last_ = 1;
    }
    // Check if the steer value indicates a negative (leftward) turn
    else if (steer < -0.1)
    {
      // Ensure the previous steering command wasn't already turning left
      if (steer_last_ != -1)
      {
        // Decrease desired curvature by a fixed step
        desired_curvature_ -= steer_btn_sign_ * steer_btn_step_;

        // Mark that steering has been updated
        steer_updated = true;
      }
      // Record the last steering direction as leftward
      steer_last_ = -1;
    }
    // If the steer value is neutral
    else
    {
      // Record the last steering direction as neutral
      steer_last_ = 0;
    }

    // If steering was updated during this cycle
    if (steer_updated)
    {
      // Round desired curvature to the nearest multiple of steer_btn_step_
      desired_curvature_ = static_cast<float>(steer_btn_step_ * round(desired_curvature_ / steer_btn_step_));

      // Clamp the desired curvature within the limits of maximum curvature
      desired_curvature_ = clamp(desired_curvature_, -max_curvature_, max_curvature_);

      // Log the updated desired curvature value
      NODELET_INFO("Desired Curvature: %f", desired_curvature_);
    }
  }
}


int SscGestureNl::getGestureVote()
{
    // Sliding window over a history of gestures to vote on the majority classification

    // Calculate the threshold for a clear majority as 75% of the history size
    int majority_threshold = static_cast<int>(0.75 * GESTURE_HISTORY_SIZE);

    // Create a map to count the occurrences of each classification in the history
    std::map<int, int> count;

    // Iterate over the gesture history
    for (int classification : gesture_history_)
    {
        // Increment the count of the current classification
        count[classification]++;
    }

    // Initialize variables to track the most frequent classification (mode) and its count
    int max_count = -1;
    int mode = -1;

    // Iterate over the counts
    for (const auto &entry : count)
    {
        // Check if the current classification's count is greater than the highest count found so far
        if (entry.second > max_count)
        {
            // Update the highest count and the mode
            max_count = entry.second;
            mode = entry.first;
        }
    }

    // If no classification has a clear majority, default to 0
    if (max_count < majority_threshold)
    {
        return 0;
    }

    // Return the classification with the highest count
    return mode;
}


void SscGestureNl::createSpeedandSteeringGesture(const ssc_joystick::Gesture::ConstPtr& msg)
{

  // Constant Speed
  const float DEFAULT_SPEED = 3; 
  // Initialize flags for speed and steering updates
  bool speed_updated = false;
  bool steering_updated = false;

  // If adding a new classification would exceed the history size, pop the oldest
  if (gesture_history_.size() >= GESTURE_HISTORY_SIZE)
  {
      gesture_history_.pop_front();
  }

  // Add new classification to history
  gesture_history_.push_back(msg->classification);

  // Use the mode from the last 20 classifications
  int consensus_classification = getGestureVote();
  NODELET_INFO("RECEIVED LABEL: %s, CLASSIFICATION: %d, CONSENSUS CLASSIFICATION: %d", msg->label.c_str(), msg->classification, consensus_classification);

  // FSM based on gesture classification
  switch(consensus_classification)
  {
    case 0: // Default: No change
      desired_velocity_ = 0.0;
      speed_updated = true;
      break;

    case 1: // Turn Left
      desired_curvature_ -= steer_btn_step_; // Decrease curvature to turn left
      steering_updated = true;
      desired_velocity_ = DEFAULT_SPEED;
      speed_updated = true;
      break;

    case 2: // Turn Right
      desired_curvature_ += steer_btn_step_; // Increase curvature to turn right
      steering_updated = true;
      desired_velocity_ = DEFAULT_SPEED;
      speed_updated = true;
      break;

    case 3: // Move Forward
      desired_velocity_ = DEFAULT_SPEED;
      speed_updated = true;
      break;

    case 4: // Stop
      desired_velocity_ = 0.0;
      speed_updated = true;
      break;

    case 5: // Slow Down
      // Assuming we're slowing down by a fixed amount (like speed_step_)
      desired_velocity_ -= speed_step_;
      if(desired_velocity_ < 0.0) // Ensure we don't go negative
      {
        desired_velocity_ = 0.0;
      }
      speed_updated = true;
      break;

    default: // For other classifications, no change
      break;
  }

  // Handle speed updates
  if(speed_updated)
  {
    // Ensure desired velocity is within limits
    if (desired_velocity_ > max_speed_)
    {
      desired_velocity_ = max_speed_;
    }
    else if (desired_velocity_ < 0.1)
    {
      desired_velocity_ = 0.0;
    }

    // Log the updated desired velocity
    NODELET_INFO("Desired velocity: %f", desired_velocity_);
  }

  // Handle steering updates
  if(steering_updated)
  {
    // Clamp the desired curvature within the limits of maximum curvature
    desired_curvature_ = clamp(desired_curvature_, -max_curvature_/10, max_curvature_/10);

    // Log the updated desired curvature value
    NODELET_INFO("Desired Curvature: %f", desired_curvature_);
  }
}


void SscGestureNl::createAuxCommand(const sensor_msgs::Joy::ConstPtr& msg)
{
  if (msg->buttons.at((uint32_t)right_turn_button_) > 0)
  {
    desired_turn_signal_ = automotive_platform_msgs::TurnSignalCommand::RIGHT;
  }
  else if (msg->buttons.at((uint32_t)left_turn_button_) > 0)
  {
    desired_turn_signal_ = automotive_platform_msgs::TurnSignalCommand::LEFT;
  }
  else
  {
    desired_turn_signal_ = automotive_platform_msgs::TurnSignalCommand::NONE;
  }
}

void SscGestureNl::diagnosticCallback(const diagnostic_msgs::DiagnosticArray::ConstPtr& msg)
{
  for (auto it = msg->status.begin(); it < msg->status.end(); it++)
  {
    if (!testing_mode_){
      if (it->name.find("Joystick Driver Status") != std::string::npos)
      {
        last_joystick_msg_timestamp_ = msg->header.stamp.toSec();
        if (it->level != diagnostic_msgs::DiagnosticStatus::OK)
        {
          NODELET_WARN("JOYSTICK FAULT");
          engaged_ = 0;
          brake_initialized_ = false;
          brake_active_ = false;
        }
      }
    }
  }
}

void SscGestureNl::gearFeedbackCallback(const automotive_platform_msgs::GearFeedback::ConstPtr& msg)
{
  current_gear_ = msg->current_gear.gear;
}

void SscGestureNl::velocityCallback(const automotive_platform_msgs::VelocityAccelCov::ConstPtr& msg)
{
  current_velocity_ = msg->velocity;
}

void SscGestureNl::inputAdasCallback(const automotive_platform_msgs::UserInputADAS::ConstPtr& msg)
{
  if (msg->btn_cc_set_inc && msg->btn_acc_gap_inc)
  {
    if (engaged_ > 0)
    {
      disengage();
    }
  }
  else if (msg->btn_cc_set_dec && msg->btn_acc_gap_dec)
  {
    if (testing_mode_){
      tryToUnsafelyEngage();
    }
    else{
      tryToEngage();
    }  
  }
}

void SscGestureNl::disengage()
{
  NODELET_INFO("Disengaged");
  engaged_ = false;
}

void SscGestureNl::tryToEngage()
{
  if (!dbw_ok_)
  {
    NODELET_INFO("Drive by wire system not ready to engage");
  }
  else if ((current_gear_ != automotive_platform_msgs::Gear::PARK) &&
           (current_gear_ != automotive_platform_msgs::Gear::NEUTRAL))
  {
    NODELET_WARN("Gear must be in park or neutral to engage");
  }
  else
  {
    NODELET_INFO("Engaged");
    desired_velocity_ = 0.0;
    desired_curvature_ = 0.0;
    desired_gear_ = current_gear_;
    engaged_ = true;
  }
}

void SscGestureNl::tryToUnsafelyEngage()
{
  NODELET_INFO("Engaged");
  desired_velocity_ = 0.0;
  desired_curvature_ = 0.0;
  desired_gear_ = current_gear_;
  engaged_ = true;
}


void SscGestureNl::moduleStateCallback(const automotive_navigation_msgs::ModuleState::ConstPtr& msg)
{
  if (msg->name == veh_controller_name_)
  {
    if (msg->state == "not_ready")
    {
      dbw_ok_ = false;
    }
    else if ((msg->state == "ready") || (msg->state == "engaged") || (msg->state == "active"))
    {
      dbw_ok_ = true;
    }
    else if (msg->state == "failure")
    {
      if (dbw_ok_ && (engaged_ > 0))
      {
        NODELET_WARN("Joystick control DISENGAGED due to %s", msg->info.c_str());
        engaged_ = 0;
      }
      dbw_ok_ = false;
    }
    else if (msg->state == "fatal")
    {
      if (dbw_ok_)
      {
        NODELET_WARN("Joystick control unavailable due to %s", msg->info.c_str());
        NODELET_WARN("Software must be stopped and restarted once the problem is fixed");
        engaged_ = 0;
      }
      dbw_ok_ = false;
    }
  }
}

void SscGestureNl::publishVehicleCommand(const ros::TimerEvent& event)
{
  (void)event;

  ros::Time current_time = ros::Time::now();
  // removed for testing 
  if (!testing_mode_){
    if (current_time.toSec() - last_joystick_msg_timestamp_ > joystick_fault_timeout_)
    {
      // Joystick has timed out
      NODELET_WARN("JOYSTICK TIMEOUT");
      last_joystick_msg_timestamp_ = current_time.toSec();
      engaged_ = 0;
    }
  }

  // NODELET_INFO("Desired Speed %f", desired_velocity_);
  automotive_platform_msgs::SpeedMode speed_cmd_msg;
  speed_cmd_msg.header.stamp = current_time;
  speed_cmd_msg.mode = engage_speed_module_ ? engaged_ : false;
  speed_cmd_msg.speed = desired_velocity_ * 0.44704f;
  speed_cmd_msg.acceleration_limit = acceleration_limit_;
  speed_cmd_msg.deceleration_limit = deceleration_;
  speed_cmd_pub_.publish(speed_cmd_msg);

  automotive_platform_msgs::SteerMode steer_cmd_msg;
  steer_cmd_msg.header.stamp = current_time;
  steer_cmd_msg.mode = engage_steering_module_ ? engaged_ : false;
  steer_cmd_msg.curvature = desired_curvature_;
  steer_cmd_msg.max_curvature_rate = max_curvature_rate_;
  steer_cmd_pub_.publish(steer_cmd_msg);

  automotive_platform_msgs::GearCommand gear_cmd_msg;
  gear_cmd_msg.command.gear = desired_gear_;
  gear_cmd_pub_.publish(gear_cmd_msg);

  automotive_platform_msgs::TurnSignalCommand turn_signal_cmd_msg;
  if (desired_turn_signal_ == automotive_platform_msgs::TurnSignalCommand::LEFT ||
      desired_turn_signal_ == automotive_platform_msgs::TurnSignalCommand::RIGHT)
  {
    turn_signal_cmd_msg.mode = 1;
  }
  else
  {
    turn_signal_cmd_msg.mode = 0;
  }
  turn_signal_cmd_msg.turn_signal = desired_turn_signal_;
  turn_signal_cmd_pub_.publish(turn_signal_cmd_msg);
}
}  // namespace astuff

PLUGINLIB_EXPORT_CLASS(astuff::SscGestureNl, nodelet::Nodelet);
