import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.activation import Sigmoid
from torch.autograd import Variable
from torchvision import transforms as trn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from turtlebot3_gazebo.scripts.decoder_net import Net

class WarehouseDataset(Dataset):
    def __init__(self):
        with open('/home/simon/catkin_ws/src/turtlebot3_gazebo/scripts/data/data_warehouse_tensors.pkl', 'rb') as f:
            self.data_set = pickle.load(f)
        self.poses = self.data_set[0]
        self.ranges = self.data_set[1]
        self.images = self.data_set[2]
    
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return {'pose': self.poses[idx], 'image': self.images[idx]}
        

def train_net(net, max_epochs=50000, batch_size=200):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)
    criterion = nn.MSELoss()

    best_test_perf = np.Inf
    epoch_train_loss = []
    epoch_test_loss = []
    for epoch in range(max_epochs):
        x, y = shuffle_unison(data.X_train, data.Y_train)
        net.train()
        batch_train_loss = 0
        for i in range(int(data.m / batch_size)):
            tensor_x = torch.from_numpy(np.float32(x[i * batch_size:(i + 1) * batch_size]))
            tensor_y = torch.from_numpy(np.float32(y[i * batch_size:(i + 1) * batch_size]))
            input_x = Variable(tensor_x)
            target_y = Variable(tensor_y, requires_grad=False)
            input_x, target_y = input_x.to(device), target_y.to(device)
            optimizer.zero_grad()  # zero the gradient buffers
            output_y = net(input_x)
            loss = criterion(output_y, target_y)
            loss.backward()
            optimizer.step()  # Does the update
            batch_train_loss += loss.item()
        scheduler.step()
        epoch_train_loss.append(batch_train_loss / batch_size)
        if epoch%10==0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, batch_train_loss / int(data.m / batch_size)))
    return epoch_train_loss, epoch_test_loss

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    net = Net()
    net.to(device)

    warehouse_dataset = WarehouseDataset()

    train_net(net, warehouse_dataset)


if __name__ == "__main__":
    main()