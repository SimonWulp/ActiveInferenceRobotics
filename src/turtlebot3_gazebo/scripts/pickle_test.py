import pickle

with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data.pkl', 'rb') as f:
    dict = pickle.load(f)
    
print(dict.keys())
