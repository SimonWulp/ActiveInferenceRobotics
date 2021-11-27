import pickle
import cv2
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse.pkl', 'rb') as f:
    data_set = pickle.load(f)

poses = torch.tensor(np.ndarray(data_set.keys()))
images = torch.tensor(np.ndarray([x[1] for x in data_set.values()]))

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_tensors.pkl', 'wb') as f:
    pickle.dump((poses, images), f)

# with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_tensors.pkl', 'rb') as f:
#     data_set = pickle.load(f)
# poses = data_set[0]
# images = data_set[1]

# for i in range(1000):
#     print('pose ', poses[i])
#     # print('image ', images[i])
#     cv2.imshow('image', images[i].numpy())
#     cv2.waitKey(3)
#     input('-')