import pickle
import cv2

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse.pkl', 'rb') as f:
    dict1 = pickle.load(f)

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_2.pkl', 'rb') as f:
    dict2 = pickle.load(f)

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_3.pkl', 'rb') as f:
    dict3 = pickle.load(f)

dict = {**dict1, **dict2, **dict3}

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_tot.pkl', 'wb') as f:
            pickle.dump(dict, f)