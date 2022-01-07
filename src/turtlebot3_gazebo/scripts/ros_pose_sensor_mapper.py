#!/usr/bin/env python3
from __future__ import print_function

import math
from typing import Set
import torch
import rospy
import cv2
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, GetModelState
from cv_bridge import CvBridge, CvBridgeError
from tf import transformations as trans
import message_filters
import pickle
import numpy as np

class PoseSensorMapper:

    def __init__(self, env):
        self.env = env

        if self.env == 'office':
            self.map_x_range = (-4.5, 3.4)
            self.map_y_range = (-3.3, 4.5)
        elif self.env == 'warehouse':
            self.map_x_range = (-4.2, 2.3)
            self.map_y_range = (-9.8, 0.8)
        elif self.env == 'outside':
            self.map_x_range = (-5.5, 6.5)
            self.map_y_range = (-5.0, 7.0)
        elif self.env == 'shapes':
            self.map_x_range = (-9.0, 9.0)
            self.map_y_range = (-9.0, 4.0)
        else:
            raise ValueError("Wrong env entered, {} is not a valid environment".format(self.env))

        self.samples_goal = 2500
        self.samples = 0

        self.pose_data = np.ndarray((0, 2), dtype='f')
        self.image_data = np.ndarray((0, 256, 256), dtype='f')
                
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.laser_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.laser_sub, self.image_sub], 10, 1)
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()

    def callback(self, laser_msg, image_msg):
        # go to new location
        next_yaw = 0.5 * np.pi
        next_x = np.round(np.random.uniform(self.map_x_range[0], self.map_x_range[1]), 3)
        next_y = np.round(np.random.uniform(self.map_y_range[0], self.map_y_range[1]), 3)
        self.update_pose(next_x, next_y, next_yaw)

        # camera image
        try:
            cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(image_msg, "mono8")[:256], (256, 256)) / 255
        except CvBridgeError as e:
            print(e)

        # add pose and observation
        self.pose_data = np.append(self.pose_data, [[next_x, next_y]], axis=0)
        self.image_data = np.append(self.image_data, [cv_image], axis=0)

        self.samples += 1

        if self.samples % 100 == 0:
            self.data = (self.pose_data, self.image_data)
            with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/shapes_fixed_mono8_2_4.pkl', 'wb') as f:
                pickle.dump(self.data, f)

            print('Pickle dumped at {} samples.'.format(self.samples))

        if self.samples >= self.samples_goal:
            self.data = (self.pose_data, self.image_data)
            with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/shapes_fixed_mono8_2_4.pkl', 'wb') as f:
                pickle.dump(self.data, f)
            
            print("Entire environment is mapped.")
            rospy.signal_shutdown('Env mapped')
        

    
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


    def legal_pos(self, x, y):
        if self._in_field(x, y) and not self._in_obj(x, y):
            return True
        else:
            return False        

    def _in_field(self, x, y):
        if x >= self.map_x_range[0] and x <= self.map_x_range[1] and y >= self.map_y_range[0] and y <= self.map_y_range[1]:
            return True
        return False

    # edit for objects in env
    def _in_obj(self, x, y):
        if self.env == 'office':
            # couch
            if x > -4.6 and x < -1.7 and y > -2.7 and y < 2.5:
                return True
            # table
            if x > -0.1 and x < 1.9 and y > -1.4 and y < 1.4:
                return True
            # corner table
            if x > -4.6 and x < -1.2 and y > 3.0 and y < 4.6:
                return True
            # corner desk
            if x > 0.6 and x < 3.6 and y > 3.0 and y < 4.6:
                return True
            return False
        
        elif self.env == 'warehouse':
            # forklift
            if x > -1.1 and x < 0.8 and y > -10 and y < -9:
                return True
            # boxes_low
            if x > -2.9 and x < -0.7 and y > -8.9 and y < -5.5:
                return True

        else:
            raise ValueError("Wrong env entered, {} is not a valid environment".format(self.env))


def main():
    env = 'shapes'
    print("Starting mapping of env {}".format(env))
    rospy.init_node('pose_sensor_mapper', anonymous=True)
    psm = PoseSensorMapper(env)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()