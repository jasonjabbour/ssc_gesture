#!/usr/bin/env python3
import os
import csv
import copy
import argparse
import itertools
import imutils
from collections import Counter
from collections import deque

import torch
import clip
import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image
from ultralytics import YOLO 

from gesture_detector.utils import CvFpsCalc
from gesture_detector.model import KeyPointClassifier
from gesture_detector.model import PointHistoryClassifier
from gesture_detector.model import MultiPointHistoryClassifier

import rospy
from ssc_joystick.msg import Gesture

KEYPOINT_CLASSIFIER_PATH = '/tmp/photondrive_ws/src/ssc_gesture/gesture_detector/model/keypoint_classifier'
KEYPOINT_HISTORY_CLASSIFIER_PATH = '/tmp/photondrive_ws/src/ssc_gesture/gesture_detector/model/point_history_classifier'
MULTI_POINT_HISTORY_CLASSIFIER_PATH = '/tmp/photondrive_ws/src/ssc_gesture/gesture_detector/model/multi_point_history_classifier'
YOLO_POSE_MODEL = 'yolov8n-pose.pt'

class PoseDetection:
    def __init__(self):

        # Pytorch Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model load
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose 
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.yolov8 = YOLO(YOLO_POSE_MODEL)
        self.load_clip_model()

        self.keypoint_classifier = KeyPointClassifier(model_path=os.path.join(KEYPOINT_CLASSIFIER_PATH, 'keypoint_classifier.tflite'))
        self.point_history_classifier = PointHistoryClassifier(model_path=os.path.join(KEYPOINT_HISTORY_CLASSIFIER_PATH, 'point_history_classifier.tflite'))
        self.multi_point_history_classifier = MultiPointHistoryClassifier(model_path=os.path.join(MULTI_POINT_HISTORY_CLASSIFIER_PATH, 'multi_point_history_classifier.tflite'), 
                                                                          score_th=.9)

        self.load_labels()

        # FPS Measurement (frames per second)
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history 
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.multi_point_history = deque(maxlen=self.history_length)

        # Left wrist id history 
        self.left_wrist_id_history = deque(maxlen=self.history_length)
        self.multi_point_id_history = deque(maxlen=self.history_length)

        # Create list for consecutive times a label id is repeated
        self.threshold = 20

        # Initialize multipoint id list
        self.most_common_multi_id = []

        # Counts the first few landmarks which exclude the leg landmarks
        self.landmark_count = 25 

        # Initialize the ROS node
        rospy.init_node('gesture_detection_node', anonymous=True)
        
        # Create a publisher. You can adjust the topic name and queue size
        self.pose_pub = rospy.Publisher('gesture_topic', Gesture, queue_size=10)

        # Example: 0 for webcam, 1 for native computer camera, 'gesture.mov' for imported video
        # self.camera_input = 0
        self.camera_input = '/tmp/photondrive_ws/src/ssc_gesture/data/motion_test_1.mov'

        if type(self.camera_input) == str:
            self.camera_input_type = 'recording'
        elif type(self.camera_input) == int:
            self.camera_input_type = 'live'

    
    def load_labels(self):

        # Read labels and creates 2 lists for keypoint and point history labels. 
        with open(os.path.join(KEYPOINT_CLASSIFIER_PATH, 'keypoint_classifier_label.csv'),
                encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(os.path.join(KEYPOINT_HISTORY_CLASSIFIER_PATH, 'point_history_classifier_label.csv'),
                encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]
        with open(os.path.join(MULTI_POINT_HISTORY_CLASSIFIER_PATH, 'multi_point_history_classifier_label.csv'),
                encoding='utf-8-sig') as f:
            self.multi_point_history_classifier_labels = csv.reader(f)
            self.multi_point_history_classifier_labels = [
                row[0] for row in self.multi_point_history_classifier_labels
            ]

    def load_clip_model(self):
        # Load the CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)


    # Set video capture input
    def set_video_input(self, camera_input):
        self.camera_input = camera_input


    def run_pose_detection(self):

        self.cap = cv.VideoCapture(self.camera_input) 
        
        self.consecutive_count = 0
        while True:
            self.fps = self.cap.get(cv.CAP_PROP_FPS)
            #self.fps = self.cvFpsCalc.get()

            # Process Key (ESC: end) 
            self.key = cv.waitKey(10)
            if self.key == 27:  # ESC
                break

            # Camera capture 
            ret, self.image = self.cap.read()
            if not ret:
                break

            # Rescale recording since too slow
            if self.camera_input_type == 'recording':
                # Set the desired width for the scaled image
                new_width = 720 

                # Resize the image while maintaining aspect ratio
                self.image = imutils.resize(self.image, width=new_width)

            # self.image = cv.flip(self.image, 1)  # Mirror display
            self.debug_image = copy.deepcopy(self.image)

            # Detection implementation 
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB) 

            self.image.flags.writeable = False
            results = self.pose.process(self.image)
            self.image.flags.writeable = True

            self.image = cv.cvtColor(self.image, cv.COLOR_RGB2BGR)

            # Use YoloV8 Pose Estimation
            yolo_pose_landmarks = self.yolov8(self.image, verbose=False)

            for result in yolo_pose_landmarks:
                boxes = result.boxes.xyxy  # Bounding box coordinates
                keypoints = result.keypoints.xy  # Keypoint coordinates
                class_ids = result.boxes.cls  # Class IDs for each detection

                # Prepare text description for CLIP
                # TODO: Subscribe to a topic with str input
                text_descriptions = ["a person in a yellow traffic vest", 
                                    "police officer", 
                                    "construction worker", 
                                    "healthcare worker", 
                                    "man dressed in a green shirt", 
                                    "dog", 
                                    "cat"]
                print("*"*20)
                for text_description in text_descriptions:

                    text_tokens = clip.tokenize([text_description]).to(self.device)

                    for idx, (box, class_id) in enumerate(zip(boxes, class_ids)):
                        if class_id == 0:  # Check if the class is 'person'
                            xmin, ymin, xmax, ymax = box

                            # Draw bounding box
                            cv.rectangle(self.image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
                            
                            # Draw label
                            label = "person"  # Assuming 'person' is the label for class_id 0
                            cv.putText(self.image, label, (int(xmin), int(ymin - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            # Draw keypoints for this person
                            for kp in keypoints:
                                for point in kp:
                                    x, y = point
                                    if xmin <= x <= xmax and ymin <= y <= ymax:  # Check if keypoint is inside the bounding box
                                        cv.circle(self.image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)

                            # Extract the region of interest (ROI) from the image based on the bounding box
                            person_roi = self.image[int(ymin):int(ymax), int(xmin):int(xmax)]

                            # Visualize the extracted ROI
                            cv.imshow(f"Person {idx} ROI", person_roi)

                            # Convert the NumPy array (OpenCV image) to a PIL Image
                            person_roi_pil = Image.fromarray(cv.cvtColor(person_roi, cv.COLOR_BGR2RGB))

                            # Preprocess the ROI and convert to a tensor
                            roi_tensor = self.clip_preprocess(person_roi_pil).unsqueeze(0).to(self.device)

                            # Calculate features with CLIP
                            with torch.no_grad():
                                image_features = self.clip_model.encode_image(roi_tensor)
                                text_features = self.clip_model.encode_text(text_tokens)

                                # Compute similarity 
                                logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
                                probs = logits_per_image.cpu().numpy()

                                print(f"{text_description} proba: {probs}")


            # Draw the pose annotation on the image.
            # self.mp_drawing.draw_landmarks(
            #     self.image,
            #     results.pose_landmarks,
            #     self.mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())   


            # image_path = '/path/to/your/image.png'
            # image = Image.open(image_path)

            # # Preprocess the image and convert to a tensor
            # image_tensor = preprocess(image).unsqueeze(0).to(device)

            # # Prepare the text
            # text_descriptions = ["a person in a yellow traffic vest"]
            # text_tokens = clip.tokenize(text_descriptions).to(device)

            # # Calculate features
            # with torch.no_grad():
            #     image_features = model.encode_image(image_tensor)
            #     text_features = model.encode_text(text_tokens)

            #     # Compute similarities between image and text features
            #     # (For multiple regions, you would repeat the image feature extraction for each ROI)
            #     logits_per_image, logits_per_text = model(image_tensor, text_tokens)
            #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # print(f"Probabilities: {probs}")


            # Create variable for previous multipoint id
            if not self.most_common_multi_id:
                previous_multipoint_id = -1
            else:
                previous_multipoint_id = self.most_common_multi_id[0][0]

            # Initialize variable 
            self.consecutive_multipoint_id = 0

            if results.pose_landmarks is not None:
                # pose_value holds x,y,z, and visibility for the 32 landmarks of each frame. 
                pose_values = getattr(results.pose_landmarks, "landmark")

                # Create blank frame list where each element in the list will hold a list with x and y values
                self.frame_list = []

                # landmark_value gives 1 of the 32 landmarks for each frame
                for i, landmark_value in enumerate(pose_values):
                    x = getattr(landmark_value, 'x')
                    y = getattr(landmark_value, 'y')
                    self.frame_list.append([x,y])

                # Bounding box calculation
                self.brect = self.calc_bounding_rect(self.debug_image, self.frame_list)

                # Landmark calculation. Landmark_list is a list for the 33 landmarks where each element is [x, y].
                self.landmark_list = self.calc_landmark_list(self.debug_image, self.frame_list)

                #Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(self.landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(self.debug_image, self.point_history)
                pre_processed_multi_point_history_list = self.pre_process_multi_point_history(self.debug_image, self.multi_point_history)

                # Pose classification
                self.keypoint_classifier = KeyPointClassifier(model_path=os.path.join(KEYPOINT_CLASSIFIER_PATH, 'keypoint_classifier.tflite'))
                self.pose_id = self.keypoint_classifier(pre_processed_landmark_list)
                self.point_history.append(self.landmark_list[16]) # for left wrist
                self.multi_point_history.append(self.landmark_list[:self.landmark_count]) # for most of body
                # Left wrist classification
                left_wrist_id = 0 # Default
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    left_wrist_id = self.point_history_classifier(
                        pre_processed_point_history_list)
                    
                # Full body classification
                full_body_id = 0 # Default
                multi_point_history_len = len(pre_processed_multi_point_history_list)
                if multi_point_history_len == (self.history_length * self.landmark_count * 2):
                    full_body_id = self.multi_point_history_classifier(
                        pre_processed_multi_point_history_list)

                # Calculates the left wrist IDs in the latest detection
                self.left_wrist_id_history.append(left_wrist_id)
                most_common_lw_id = Counter(
                    self.left_wrist_id_history).most_common()
                
                # Calculates the full body IDs in the latest detection
                self.multi_point_id_history.append(full_body_id)
                self.most_common_multi_id = Counter(
                    self.multi_point_id_history).most_common()        

                if self.most_common_multi_id == 4:
                    print("move forward")
                # Ensure multipoint id lasts for longer    
                if self.most_common_multi_id[0][0] == previous_multipoint_id:
                    self.consecutive_count += 1
                    if self.consecutive_count >= self.threshold:
                        self.consecutive_multipoint_id = self.most_common_multi_id[0][0]
                    else:
                        self.consecutive_multipoint_id = 0
                else:
                    self.consecutive_count = 0
                

                # Add classification to image
                self.image = self.draw_info_text(
                    self.image,
                    self.brect,
                    self.keypoint_classifier_labels[self.pose_id],
                    self.consecutive_count,
                    self.multi_point_history_classifier_labels[self.consecutive_multipoint_id]
                    )

                # Publish the gesture
                gesture_msg = Gesture()

                gesture_msg.classification = self.consecutive_multipoint_id
                gesture_msg.label = self.multi_point_history_classifier_labels[self.consecutive_multipoint_id]

                self.pose_pub.publish(gesture_msg)
                
            else:
                self.point_history.append([0, 0])
                default_coordinates = [0,0]
                multi_default_coordinates = [default_coordinates] * self.landmark_count
                self.multi_point_history.append(multi_default_coordinates)

            self.image = self.draw_info(self.image, self.fps)

            # Screen reflection 
            cv.imshow('MediaPipe Pose', self.image)

        self.cap.release()
        cv.destroyAllWindows()


    # With alterations this function will need to take in a 2D list in the place of second argument
    def calc_bounding_rect(self, image, frame):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(frame):
            landmark_x = min(int(landmark[0] * image_width), image_width - 1)
            landmark_y = min(int(landmark[1] * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    # With alterations this function will need to take in a 2D list in the place of second argument
    def calc_landmark_list(self, image, frame):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(frame):
            landmark_x = min(int(landmark[0] * image_width), image_width - 1)
            landmark_y = min(int(landmark[1] * image_height), image_height - 1)
            # landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history


    def pre_process_multi_point_history(self, image, multi_point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_multi_point_history = copy.deepcopy(multi_point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, multi_point in enumerate(temp_multi_point_history):
            # multi_point is a 2D list of multiple landmarks at a given time
            # Create another for loop to go through each landmark
            for landmark_id, landmark_point in enumerate(multi_point):
                if index == 0:
                    base_x, base_y = landmark_point[0], landmark_point[1]
                multi_point[landmark_id][0] = (multi_point[landmark_id][0] - 
                                            base_x) / image_width
                multi_point[landmark_id][1] = (multi_point[landmark_id][1] - 
                                            base_y) / image_height
                
            # Convert to a one-dimensional list
            multi_point = list(
            itertools.chain.from_iterable(multi_point))
            temp_multi_point_history[index] = multi_point

        # Convert to a one-dimensional list
        temp_multi_point_history = list(
            itertools.chain.from_iterable(temp_multi_point_history))
        
        return temp_multi_point_history


    def draw_info_text(self, image, brect, pose_class_text, consecutive_count, multi_point_text):
        if pose_class_text != "":
            cv.putText(image, "Classification:" + pose_class_text, (10, 70),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv.LINE_AA)      
        if consecutive_count != "":
            cv.putText(image, "Number of repetitions for current classification:" + str(consecutive_count), (10, 105),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv.LINE_AA)
        if multi_point_text != "":
            cv.putText(image, "Full Body Motion:" + multi_point_text, (10, 140),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv.LINE_AA)
        return image


    # Adds circles on video to represent each landmark. 
    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                        (152, 251, 152), 2)

        return image


    # Adds logging information to video
    def draw_info(self, image, fps):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)

        return image



if __name__ == '__main__':
    pose_detection = PoseDetection()
    pose_detection.run_pose_detection()
