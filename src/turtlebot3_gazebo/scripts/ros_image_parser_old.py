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
import os
from PIL import Image as ImagePIL

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw_throttle",Image,self.callback)
    self.scene_classifier = SceneClassifier()

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as e:
      print(e)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    pil_image = ImagePIL.fromarray(cv_image)

    input_img = V(self.scene_classifier.centre_crop(pil_image).unsqueeze(0))

    # forward pass
    logit = self.scene_classifier.model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('{:.3f} -> {}'.format(probs[0], self.scene_classifier.classes[idx[0]]))

class SceneClassifier:

  def __init__(self):

    # th architecture to use
    self.arch = 'resnet18'

    # load the pre-trained weights
    self.model_file = '%s_places365.pth.tar' % self.arch
    if not os.access(self.model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + self.model_file
        os.system('wget ' + weight_url)

    self.model = models.__dict__[self.arch](num_classes=365)
    self.checkpoint = torch.load(self.model_file, map_location=lambda storage, loc: storage)
    self.state_dict = {str.replace(k,'module.',''): v for k,v in self.checkpoint['state_dict'].items()}
    self.model.load_state_dict(self.state_dict)
    self.model.eval()

    # load the image transformer
    self.centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    self.file_name = 'categories_places365.txt'
    if not os.access(self.file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
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