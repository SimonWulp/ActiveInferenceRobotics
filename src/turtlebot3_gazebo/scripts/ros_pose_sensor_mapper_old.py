#!/usr/bin/env python3
from __future__ import print_function

import sys
import math
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.srv import SetModelState, GetModelState
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as ImagePIL
from tf import transformations as trans
import message_filters

class PoseSensorMapper:

    def __init__(self):
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.laser_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.laser_sub, self.image_sub], 10, 1)
        self.ts.registerCallback(self.callback)
        
        self.bridge = CvBridge()

    def callback(self, laser_msg, image_msg):
        # ##POSE

        ros_pose = self.get_state('turtlebot3_waffle_pi', '').pose

        print(round(ros_pose.position.x, 2), '\t', round(ros_pose.position.y, 2))

        # # position coordinates
        # ros_pos = pose_msg.pose.pose.position
        # ros_x = ros_pos.x
        # ros_y = ros_pos.y

        # # orientation quaternion to yaw in 0-360 degrees
        # ros_ori_quat = pose_msg.pose.pose.orientation
        # ros_ori_euler = trans.euler_from_quaternion([ros_ori_quat.x, ros_ori_quat.y, ros_ori_quat.z, ros_ori_quat.w])
        # ros_ori_yaw_deg = abs(math.degrees(ros_ori_euler[2]) % 360)

        # # pose in (x-coordinate, y-coordinate, yaw in degrees)
        # ros_pose = (ros_x, ros_y, ros_ori_yaw_deg)

        # ## LASER

        # # laser ranges
        # ranges = laser_msg.ranges

        # ## IMAGE
        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # except CvBridgeError as e:
        #     print(e)

        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(3)


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