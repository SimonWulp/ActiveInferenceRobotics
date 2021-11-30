from torch import nn
from torch.utils.data import Dataset
import pickle

# https://github.com/pl-robotdecision/pixel-aif/blob/master/nao_simulation/catkin_ws/src/my_executables/scripts/Conv_decoder_model/conv_decoder.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ff_layers = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 20 * 15 * 8), # 2400 neurons
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            # state size. 8 x 20 x 15
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            # state size. 16 x 40 x 30
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # state size. 64 x 80 x 60
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            # state size. 16 x 160 x 120
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
            # state size. 1 x 320 x 240
        )

    def forward(self, x):
        x = self.ff_layers(x)
        x = x.view(-1, 8, 20, 15)
        x = self.conv_layers(x)
        return x

class WarehouseDatasetImages:
    def __init__(self):
        with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_tensors.pkl', 'rb') as f:
            self.data_set = pickle.load(f)

        self.X_train = self.data_set[0]
        # self.ranges = self.data_set[1]
        self.Y_train = self.data_set[2]
        self.Y_train = self.Y_train.reshape([-1, 1, self.Y_train.shape[2], self.Y_train.shape[1]])

        self.length = self.X_train.shape[0]

class OfficeDatasetImages:
    def __init__(self):
        with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_office_tensors.pkl', 'rb') as f:
            self.data_set = pickle.load(f)

        self.X_train = self.data_set[0]
        # self.ranges = self.data_set[1]
        self.Y_train = self.data_set[2]
        self.Y_train = self.Y_train.reshape([-1, 1, self.Y_train.shape[2], self.Y_train.shape[1]])

        self.length = self.X_train.shape[0]