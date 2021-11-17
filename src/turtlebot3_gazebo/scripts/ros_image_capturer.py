#!/usr/bin/env python3
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as ImagePIL

class image_converter:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle",Image,self.callback)
        self.img_index = 0

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image window", cv_image)

        dir_path = '/home/simon/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/scripts/data/office/'
        img_name = 'office.' + str(self.img_index) + '.jpg'
        full_path = dir_path + img_name

        cv2.imwrite(full_path, cv_image)

        print('img ', self.img_index, ' written')

        if self.img_index > 5000:
            sys.exit(0)

        self.img_index+=1

        cv2.waitKey(3)

    

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)