#!/usr/bin/env python3
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from torch import nn
import os
from PIL import Image as ImagePIL

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle",Image,self.callback)
    self.scene_classifier = SceneClassifier()

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    pil_image = ImagePIL.fromarray(cv_image)
    input_img = V(self.scene_classifier.transform(pil_image).unsqueeze(0))

    # forward pass
    logit = self.scene_classifier.model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('{:.3f} -> {}'.format(probs[0], self.scene_classifier.classes[idx[0]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SceneClassifier:

  def __init__(self):
    self.model = Model()    
    self.model.load_state_dict(torch.load('/home/simon/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/scripts/ros_model.pth'))
    self.model.eval()

    # load the image transformer
    self.transform = trn.Compose([
                trn.RandomResizedCrop(128),
                trn.ToTensor()])

    # load the class label
    self.file_name = '/home/simon/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/scripts/categories_ros.txt'
    self.classes = list()
    with open(self.file_name) as class_file:
        for line in class_file:
            self.classes.append(line.strip().split(' ')[0][3:])
    self.classes = tuple(self.classes)


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