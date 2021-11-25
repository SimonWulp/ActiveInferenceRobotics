import pickle
import cv2

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse.pkl', 'rb') as f:
    dict = pickle.load(f)

print(len(dict))
    
ls = list(dict.values())

ls_images = [l[1] for l in ls]


cv2.imshow('picture', ls_images[30])
cv2.waitKey()

# for i, im in enumerate(ls_images):
#     cv2.imwrite(str(i) + '.jpg', im)
#     cv2.waitKey(3)