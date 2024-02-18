#!/usr/bin/env python3

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import rospy
from sensor_msgs.msg import Image as ROSImage
import cv2
from cv_bridge import CvBridge

class PassengerDashboardApp(App):
    def __init__(self, **kwargs):
        # Call the superclass initializer
        super(PassengerDashboardApp, self).__init__(**kwargs)  
        
        # Create an Image widget for displaying images
        self.img = Image()  
        # Create a TextInput widget for user input with a hint, set its height, and disable size adjustment
        self.description_input = TextInput(hint_text='Describe the person to follow', size_hint_y=None, height=50)

        # Initialize the ROS node with a unique name for this subscriber
        rospy.init_node('passenger_dashboard_node', anonymous=True)
        # Subscribe to the 'llava_image_topic' ROS topic and specify the callback method
        rospy.Subscriber('llava_image_topic', ROSImage, self.image_format_callback)
        # Initialize a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()  

    def build(self):
        # Create a vertical BoxLayout for the widgets
        layout = BoxLayout(orientation='vertical') 
        # Add the TextInput widget to the layout 
        layout.add_widget(self.description_input)  
        # Add the Image widget to the layout
        layout.add_widget(self.img)  
        # Return the layout as the root widget of the app
        return layout  

    def image_format_callback(self, data):
        # Use CvBridge to convert the incoming ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # Convert the OpenCV image from BGR to RGB color space
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Schedule the update of the image on the main thread
        Clock.schedule_once(lambda dt: self.update_image_with_cv_image(cv_image))

    def update_image_with_cv_image(self, cv_image):
        # Build a texture from the OpenCV image
        texture = self.build_texture(cv_image)
        # Update the texture of the Image widget with the new texture
        self.img.texture = texture

    def build_texture(self, cv_image):
        # Create a new Texture object with the size and color format matching the OpenCV image
        texture = Texture.create(size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='rgb')
        # Update the texture with the OpenCV image data
        texture.blit_buffer(cv_image.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        # Return the created texture
        return texture  

if __name__ == '__main__':
    # Create an instance of the app
    app = PassengerDashboardApp()  
    # Start the app
    app.run()  
