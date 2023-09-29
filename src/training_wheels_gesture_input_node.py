#!/usr/bin/env python3

import rospy
from ssc_joystick.msg import Gesture
from collections import deque
import threading

class TrainingWheelsGestureDetector:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gesture_detection_node', anonymous=True)
        
        # Create a publisher. You can adjust the topic name and queue size
        self.gesture_pub = rospy.Publisher('training_wheels_gesture_topic', Gesture, queue_size=10)

        # Queue to store the last 20 classifications
        self.gesture_queue = deque(maxlen=20)

        # List of gesture labels
        self.gesture_labels = ["Default", "Turn Left", "Turn Right", "Move Forward", "Stop", "Slow Down"]

        # Start input collection in a separate thread
        input_thread = threading.Thread(target=self.collectInput)
        input_thread.start()

        self.publishGestures()

    def collectInput(self):
        while not rospy.is_shutdown():
            # Ask for user input
            user_input = input("Enter gesture classification (0-5, or multiple separated by space): ").strip()

            # Split input based on spaces and extend the queue
            gestures = list(map(int, user_input.split()))
            for g in gestures:
                if 0 <= g <= 5:
                    self.gesture_queue.append(g)
                else:
                    print("Invalid gesture classification. Please enter a value between 0 and 5.")

    def publishGestures(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Publish gestures one by one from the queue
            for g in self.gesture_queue:
                gesture_msg = Gesture()
                gesture_msg.classification = g
                gesture_msg.label = self.gesture_labels[g]
                self.gesture_pub.publish(gesture_msg)
            rate.sleep()

if __name__ == "__main__":
    try:
        detector = TrainingWheelsGestureDetector()
    except rospy.ROSInterruptException:
        pass
