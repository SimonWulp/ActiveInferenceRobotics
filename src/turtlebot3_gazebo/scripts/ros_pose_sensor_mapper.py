#!/usr/bin/env python3
from __future__ import print_function

import sys
import math
from typing import Set
from genpy import message
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState, GetModelState
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as ImagePIL
from tf import transformations as trans
import message_filters
import pickle

mapping = {}

class PoseSensorMapper:

    def __init__(self):
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.map_x_range = (-4.5, 3.5)
        self.map_y_range = (-3.5, 3.5)

        self.x = self.map_x_range[0]
        self.y = self.map_y_range[0]
        self.step_size = 3

        self.turning_mode = True
        self.turns = 16
        
        self.update_pose(self.x, self.y, 0)

        self.laser_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.laser_sub, self.image_sub], 10, 1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

    def callback(self, laser_msg, image_msg):
        callback_ended = True     
        # laser ranges
        ranges = laser_msg.ranges
        # print(ranges)

        # camera image
        try:
            cv_image = cv2.resize(self.bridge.imgmsg_to_cv2(image_msg, "mono8"), (320, 240), interpolation=cv2.INTER_AREA)
            # cv_image = self.bridge.imgmsg_to_cv2(image_msg, "mono8")
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        ros_pose = self.get_state('turtlebot3_waffle_pi', '').pose
        ros_position = ros_pose.position
        ros_orientation = ros_pose.orientation

        if not self.turning_mode:
            if self.x + self.step_size <= self.map_x_range[1]:
                next_x = self.x + self.step_size
                next_y = self.y
                if self.legal_pos(next_x, next_y):
                    self.update_pose(next_x, next_y, 0)
                    self.turning_mode = True
                    callback_ended = False
                    print('Robot moved to ', next_x, '\t', next_y)
                self.x = next_x
                self.y = next_y

            elif self.y + self.step_size <= self.map_y_range[1]:
                next_x = self.map_x_range[0]
                next_y = self.y + self.step_size
                if self.legal_pos(next_x, next_y):
                    self.update_pose(next_x, next_y, 0)
                    self.turning_mode = True
                    callback_ended = False
                    print('Robot moved to ', next_x, '\t', next_y)
                self.x = next_x
                self.y = next_y

            else:
                rospy.signal_shutdown('Field mapped')
        

        if self.turning_mode and callback_ended and self.x == round(ros_position.x, 2) and self.y == round(ros_position.y, 2):
            if self.turns > 0:
                print('Picture taken at {}\t{}\t{}'.format(self.x, self.y, self.turns*22.5))
                # im_str = '/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/' + str(self.x) + '_' + str(self.y) + '_' + str((self.turns*22.5)%360) + '.jpg'
                mapping[(self.x, self.y, self.turns*22.5)] = (ranges, cv_image)
                self.update_pose(self.x, self.y, (self.turns-1)*22.5)
                self.turns -= 1
                
                
            if self.turns == 0:
                self.turning_mode = False
                self.turns = 16


        
    def legal_pos(self, x, y):
        if self._in_field(x, y) and not self._in_couch(x, y) and not self._in_table(x, y):
            return True
        else:
            return False

    def _in_field(self, x, y):
        if x >= self.map_x_range[0] and x <= self.map_x_range[1] and y >= self.map_y_range[0] and y <= self.map_y_range[1]:
            return True
        return False

    def _in_couch(self, x, y):
        if x > -4.6 and x < -2 and y > -2.7 and y < 2.2:
            return True
        return False

    def _in_table(self, x, y):
        if x > 0 and x < 1.7 and y > -1.2 and y < 1.3:
            return True
        return False

    def update_pose(self, x, y, yaw_degrees):
        next_state = ModelState()
        next_state.model_name = 'turtlebot3_waffle_pi'

        next_state.pose.position.x = x
        next_state.pose.position.y = y
        
        quaternion = self.yaw_to_quat(yaw_degrees)
        next_state.pose.orientation.x = quaternion[0]
        next_state.pose.orientation.y = quaternion[1]
        next_state.pose.orientation.z = quaternion[2]
        next_state.pose.orientation.w = quaternion[3]

        self.set_state(next_state)

    def quat_to_yaw(self, quaternion):
        ros_ori_euler = trans.euler_from_quaternion(quaternion)
        return abs(math.degrees(ros_ori_euler[2]) % 360)

    def yaw_to_quat(self, yaw_degrees):
        yaw_radians = math.radians(yaw_degrees)
        return trans.quaternion_from_euler(0, 0, yaw_radians)


def main():
    rospy.init_node('pose_sensor_mapper', anonymous=True)
    psm = PoseSensorMapper()

    try:
        rospy.spin()
        print('Pickle dumped')
        with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data.pkl', 'wb') as f:
            pickle.dump(mapping, f)

    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()