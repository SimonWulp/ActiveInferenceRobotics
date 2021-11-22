#!/usr/bin/env python3
from __future__ import print_function

import sys
import math
from typing import Set
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

class PoseSensorMapper:

    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
           self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException :
            print("Service call failed")

        self.state_msg = ModelState()
        self.state_msg.model_name = 'turtlebot3_waffle_pi'

        self.laser_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.laser_sub, self.image_sub], 10, 1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

        self.i = 0
        self.o = 0

    def callback(self, laser_msg, image_msg):
        # position
        ros_pos = self.state_msg.pose.position

        # laser ranges
        ranges = laser_msg.ranges

        # # camera image
        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # except CvBridgeError as e:
        #     print(e)

        # cv2.imshow("Image window", cv_image)

        self.update_pose(-self.i, -self.i, self.o)

        print(ros_pos.x, ros_pos.y)

        cv2.waitKey(3)

        input('Press key')

        # self.i += 1
        self.o += 45

    def update_pose(self, x, y, yaw_degrees):
        self.state_msg.pose.position.x = x
        self.state_msg.pose.position.y = y
        
        quaternion = self.yaw_to_quat(yaw_degrees)
        self.state_msg.pose.orientation.x = quaternion[0]
        self.state_msg.pose.orientation.y = quaternion[1]
        self.state_msg.pose.orientation.z = quaternion[2]
        self.state_msg.pose.orientation.w = quaternion[3]

        self.set_state(self.state_msg)

    def quat_to_yaw(quaternion):
        ...
        # ros_ori_quat = pose_msg.pose.pose.orientation
        # ros_ori_euler = trans.euler_from_quaternion([ros_ori_quat.x, ros_ori_quat.y, ros_ori_quat.z, ros_ori_quat.w])
        # ros_ori_yaw_deg = abs(math.degrees(ros_ori_euler[2]) % 360)

    def yaw_to_quat(self, yaw_degrees):
        yaw_radians = math.radians(yaw_degrees)
        return trans.quaternion_from_euler(0, 0, yaw_radians)


def main():
    rospy.init_node('pose_sensor_mapper', anonymous=True)
    psm = PoseSensorMapper()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()