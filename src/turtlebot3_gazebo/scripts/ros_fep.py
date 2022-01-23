#!/usr/bin/env python3
from __future__ import print_function

import torch
import rospy
import cv2
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, GetModelState
from cv_bridge import CvBridge
from tf import transformations as trans
from torchvision import transforms
import message_filters
from fep import FEP
import numpy as np
from nets import Classifier, ConvDecoder

class RosFEP:

    def __init__(self, env):
        self.env = env

        if self.env == 'outside':
            self.map_x_range = (-6.0, 6.0)
            self.map_y_range = (-6.0, 6.0)
        elif self.env == 'inside':
            self.map_x_range = (-3.0, 3.0)
            self.map_y_range = (-3.0, 3.0)
        elif self.env == 'shapes':
            self.map_x_range = (-8.0, 8.0)
            self.map_y_range = (-8.0, 8.0)
        else:
            raise ValueError("Wrong env entered, {} is not a valid environment".format(self.env))

        self.x = self.y = 0
        self.yaw = 0.5 * np.pi
                
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.update_pose(self.x, self.y, self.yaw)

        self.laser_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.laser_sub, self.image_sub], 10, 1)
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()

        self.tf = transforms.Resize((80, 80))

        self.decoder = ConvDecoder()
        self.decoder.cpu()
        self.decoder.load_state_dict(torch.load("PATH"))
        self.decoder.eval()        

        self.classifier = Classifier()
        self.classifier.cpu()
        self.classifier.load_state_dict(torch.load("PATH"))
        self.classifier.eval()

        self.fep = FEP(self.decoder, self.classifier)


    def callback(self, laser_msg, image_msg):
        # pose
        pose = (self.x, self.y, self.yaw)
        
        # image
        np_image = cv2.resize(self.bridge.imgmsg_to_cv2(image_msg, "rgb8")[:256], (256, 256)) / 255
        torch_image = torch.tensor(np_image.transpose(2,0,1)).unsqueeze(0)
        resized_image = self.tf(torch_image).squeeze().numpy()

        # fep
        # TODO

        # go to new location
        self.yaw = np.random.uniform(-np.pi, np.pi)
        self.x = np.round(np.random.uniform(self.map_x_range[0], self.map_x_range[1]), 3)
        self.y = np.round(np.random.uniform(self.map_y_range[0], self.map_y_range[1]), 3)
        self.update_pose(self.x, self.y, self.yaw)
        
    
    def update_pose(self, x, y, yaw):
        next_state = ModelState()
        next_state.model_name = 'turtlebot3_waffle_pi'

        next_state.pose.position.x = x
        next_state.pose.position.y = y
        
        quaternion = trans.quaternion_from_euler(0, 0, yaw)
        next_state.pose.orientation.x = quaternion[0]
        next_state.pose.orientation.y = quaternion[1]
        next_state.pose.orientation.z = quaternion[2]
        next_state.pose.orientation.w = quaternion[3]

        self.set_state(next_state)

    def select_env():
        ...


def main():
    env = 'shapes'
    print("Starting simulation of env {}".format(env))
    rospy.init_node('ros_fep', anonymous=True)
    rfep = RosFEP(env)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()