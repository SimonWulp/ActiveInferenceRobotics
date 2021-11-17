#!/usr/bin/env python3
from __future__ import print_function

import sys
import math
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as ImagePIL
from tf import transformations as trans

class PoseSensorMapper:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle", Image, self.callback)
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.callback)

    def callback(self,data):
        # position coordinates
        ros_pos = data.pose.pose.position
        ros_x = ros_pos.x
        ros_y = ros_pos.y

        # orientation quaternion to yaw in 0-360 degrees
        ros_ori_quat = data.pose.pose.orientation
        ros_ori_euler = trans.euler_from_quaternion([ros_ori_quat.x, ros_ori_quat.y, ros_ori_quat.z, ros_ori_quat.w])
        ros_ori_yaw_deg = abs(math.degrees(ros_ori_euler[2]) % 360)

        # pose in (x-coordinate, y-coordinate, yaw in degrees)
        ros_pose = (ros_x, ros_y, ros_ori_yaw_deg)

        print('{:.2f},\n'.format(ros_ori_yaw_deg))
        

    

def main(args):
    psm = PoseSensorMapper()
    rospy.init_node('PoseSensorMapper', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)