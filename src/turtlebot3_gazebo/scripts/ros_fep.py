#!/usr/bin/env python3
from __future__ import print_function
from tracemalloc import start

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
import matplotlib.pyplot as plt
import pickle
from nets import Classifier, ConvDecoder

class RosFEP:

    def __init__(self, env, start_pose, max_steps, active_inference, attractor_img=None):
        self.env = env
        self.start_pose = start_pose
        self.max_steps = max_steps
        self.active_inference = active_inference
        if self.active_inference:
            if attractor_img is None:
                raise ValueError("Give an attractor image to use for active inference.")
            else:
                self.attractor_img = attractor_img

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

        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.steps_taken = 0
        self.x, self.y = self.start_pose               
        self.update_pose(self.x, self.y)

        if not self.active_inference:
            self.x = np.round(np.random.uniform(self.map_x_range[0], self.map_x_range[1]), 3)
            self.y = np.round(np.random.uniform(self.map_y_range[0], self.map_y_range[1]), 3)
            self.update_pose(self.x, self.y)

        """FEP"""
        self.decoder1 = ConvDecoder()
        self.decoder1.cpu()
        self.decoder1.load_state_dict(torch.load("/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/trained_conv_nets/deconv_shapes_fixed_2_200e_USE.pt"))
        self.decoder1.eval()

        self.decoder2 = ConvDecoder()
        self.decoder2.cpu()
        self.decoder2.load_state_dict(torch.load("/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/trained_conv_nets/deconv_shapes_dif_fixed_1_200e_USE.pt"))
        self.decoder2.eval()

        self.ls_decoders = [self.decoder1, self.decoder2]

        self.classifier = Classifier()
        self.classifier.cpu()
        self.classifier.load_state_dict(torch.load("/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/trained_conv_nets/classifier_2_100e_USE.pt"))
        self.classifier.eval()
        self.classes = ['squares', 'circles']

        self.fep = FEP(self.ls_decoders, self.classifier, self.active_inference)
        self.fep.mu = self.norm_pose(self.fep.mu.squeeze())
        self.fep.a = self.norm_pose(self.fep.a.squeeze())

        if self.active_inference:
            self.fep.attractor_image = self.attractor_img
        """FEP"""

        """ROS HANDLING"""
        self.laser_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.laser_sub, self.image_sub], 10, 1)
        self.ts.registerCallback(self.callback)
        """ROS HANDLING"""

        """IMAGE PROCESSING"""
        self.bridge = CvBridge()
        self.tf = transforms.Resize((80, 80))
        """IMAGE PROCCESSING"""


    def callback(self, laser_msg, image_msg):
        # pose
        self.pose = [self.x, self.y]
        
        # image
        np_image = cv2.resize(self.bridge.imgmsg_to_cv2(image_msg, "rgb8")[:256], (256, 256)) / 255
        torch_image = torch.tensor(np_image.transpose(2,0,1)).unsqueeze(0)
        resized_image = self.tf(torch_image).squeeze().numpy()

        # fep
        self.fep.s_v = resized_image

        self.fep.step()

        if self.active_inference:
            print(self.unnorm_pose(self.fep.a.squeeze()))
            self.x, self.y = self.unnorm_pose(self.fep.a.squeeze())

            # go to new location
            self.update_pose(self.x, self.y)

        # max steps
        self.steps_taken += 1
        if self.steps_taken > self.max_steps:
            rospy.signal_shutdown('max steps reached')

    
    def update_pose(self, x, y, yaw=0.5*np.pi):
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


    def select_env(self):
        # TODO: context switching
        ...

    def norm_pose(self, pose):
        x = (pose[0] - self.map_x_range[0]) / (self.map_x_range[1] - self.map_x_range[0])
        y = (pose[1] - self.map_y_range[0]) / (self.map_y_range[1] - self.map_y_range[0])

        return (x, y)

    def unnorm_pose(self, pose):
        x = pose[0] * (self.map_x_range[1] - self.map_x_range[0]) + self.map_x_range[0]
        y = pose[1] * (self.map_y_range[1] - self.map_y_range[0]) + self.map_y_range[0]

        return [x, y]


def main():
    env = 'shapes'
    print("Starting simulation of env {}".format(env))
    rospy.init_node('ros_fep', anonymous=True)

    start_pose = [0., 0.]
    active_inference = True
    max_steps = 150
   
    if active_inference:
        # get random attractor image
        with open("/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/shapes_random_samples_100.pkl", 'rb') as f:
            pose_samples, img_samples = pickle.load(f)
        sample_idx = 50
        att_img = img_samples[sample_idx]

        plt.imshow(att_img.transpose(1,2,0))
        plt.show()

        rfep = RosFEP(env, start_pose, max_steps, active_inference, att_img)

        print(rfep.unnorm_pose(pose_samples[sample_idx]))

        rospy.spin()

        goal_pose = rfep.unnorm_pose(pose_samples[sample_idx])
        end_pose = rfep.unnorm_pose(rfep.fep.a_hist[-1])
        dist = np.sqrt((goal_pose[0] - end_pose[0])**2 + (goal_pose[1] - end_pose[1])**2)

        print('goal pos: [{:.3f}, {:.3f}], end pos: [{:.3f}, {:.3f}]'.format(goal_pose[0], goal_pose[1], end_pose[0], end_pose[1]))
        print('Max steps reached, distance from goal: {:.3f}'.format(dist))

    else:
        rfep = RosFEP(env, start_pose, max_steps, active_inference)
        rospy.spin()

        goal_pose = rfep.pose
        end_pose = rfep.unnorm_pose(rfep.fep.mu_hist[-1])
        dist = np.sqrt((goal_pose[0] - end_pose[0])**2 + (goal_pose[1] - end_pose[1])**2)

        print('goal pos: [{:.3f}, {:.3f}], end pos: [{:.3f}, {:.3f}]'.format(goal_pose[0], goal_pose[1], end_pose[0], end_pose[1]))
        print('Max steps reached, distance from goal: {:.3f}'.format(dist))

if __name__ == '__main__':
    main()