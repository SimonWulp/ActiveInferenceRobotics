import torch
from torch import nn
from torch._C import Value
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import os
import numpy as np

# https://github.com/pl-robotdecision/pixel-aif/blob/master/nao_simulation/catkin_ws/src/my_executables/scripts/Conv_decoder_model/conv_decoder.py
class ImageNet(nn.Module):

    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_conv_nets/")

    def __init__(self):
        super(ImageNet, self).__init__()
        self.ff_layers = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 8 * 15 * 8), # 2400 neurons
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            # state size. 8 x 15 x 8
            nn.ConvTranspose2d(8, 16, (3, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            # state size. 16 x 30 x 15
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            # state size. 64 x 60 x 30
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            # state size. 16 x 120 x 60
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
            # state size. 1 x 240 x 120
        )

    def forward(self, x):
        x = self.ff_layers(x)
        x = x.view(-1, 8, 8, 15)
        x = self.conv_layers(x)
        return x

    @staticmethod
    def train_net(net, loader, device, net_id, max_epochs=10):
        print("Starting training.")
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.95)
        criterion = nn.MSELoss()

        epoch_loss = []
        # val_loss = []
        batch_loss = []

        for epoch in range(max_epochs):
            cur_batch_loss = []
            net.train()
            for i, (pos, img)  in enumerate(loader):
                pos = pos.unsqueeze(1).to(device).to(torch.float32)
                img = img.unsqueeze(1).to(device).to(torch.float32)

                optimizer.zero_grad()  # zero the gradient buffers
                output = net(pos)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()  # Does the update
                loss = loss.item()

                cur_batch_loss = np.append(cur_batch_loss, [loss])
                batch_loss = np.append(batch_loss, [loss])

            # scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            # cur_val_loss = Net.eval_test_set(net, data, device, optimizer, criterion)
            # val_loss = np.append(val_loss, [np.mean(cur_val_loss)])
            
            if epoch%1==0:
                print('------ Epoch ', epoch)
                print('Epoch loss:', epoch_loss[-1])
                # print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(), ImageNet.SAVE_PATH + "img/" + str(net_id) + ".pt")
        
        torch.save(net.state_dict(), ImageNet.SAVE_PATH + "img/" + str(net_id) + ".pt")

        return epoch_loss, batch_loss

    # @staticmethod
    # def eval_test_set(net, data, device, optimizer, criterion):
    #     net.eval()
    #     cur_val_loss = []
    #     x, y = DataSet.shuffle_unison(data.X_test, data.Y_img_test)
    #     for i in range(data.X_test.shape[0]):
    #         input_x = torch.tensor(np.float32(x[i:(i + 1)]), device=device)
    #         target_y = torch.tensor(np.float32(y[i:(i + 1)]), device=device)

    #         optimizer.zero_grad()  # zero the gradient buffers
    #         output_y = net(input_x)
    #         loss = criterion(output_y, target_y)
    #         cur_val_loss = np.append(cur_val_loss, [loss])

    #     return np.mean(cur_val_loss)

class RangeNet(nn.Module):

    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_conv_nets/")

    def __init__(self):
        super(RangeNet, self).__init__()
        self.ff_layers = nn.Sequential(
            nn.Linear(3, 45),
            nn.ReLU(),
            nn.Linear(45, 90),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            # shape 1 * 180
            nn.ConvTranspose1d(1, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 1, stride=1, padding=0),
            nn.ReLU(),
            # shape 1 * 360
            nn.ConvTranspose1d(1, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 1, stride=1, padding=0),
            nn.ReLU()
            # shape 1 * 720
        )


    def forward(self, x):
        x = self.ff_layers(x)
        x = x.view(-1, 1, 90)
        x = self.conv_layers(x)
        return x


    @staticmethod
    def train_net(net, loader, device, net_id, max_epochs=100):
        print("Starting training.")
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # scheduler = StepLR(optimizer, step_size=2000, gamma=0.95)
        criterion = nn.L1Loss()

        epoch_loss = []
        # val_loss = []
        batch_loss = []

        for epoch in range(max_epochs):
            cur_batch_loss = []
            net.train()
            for i, (pos, rng)  in enumerate(loader):
                pos = pos.unsqueeze(1).to(device)
                rng = rng.unsqueeze(1).to(device)

                optimizer.zero_grad()  # zero the gradient buffers
                output = net(pos)
                loss = criterion(output, rng)
                loss.backward()
                optimizer.step()  # Does the update
                loss = loss.item()
                
                cur_batch_loss = np.append(cur_batch_loss, [loss])
                batch_loss = np.append(batch_loss, [loss])

            # scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            # cur_val_loss = Net.eval_test_set(net, data, device, optimizer, criterion)
            # val_loss = np.append(val_loss, [np.mean(cur_val_loss)])
            
            if epoch%10==0:
                print('------ Epoch ', epoch)
                print('Epoch loss:', epoch_loss[-1])
                # print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(), RangeNet.SAVE_PATH + "rng/" + str(net_id) + ".pt")
        
        torch.save(net.state_dict(), RangeNet.SAVE_PATH + "rng/" + str(net_id) + ".pt")

        return epoch_loss, batch_loss

    # @staticmethod
    # def eval_test_set(net, data, device, optimizer, criterion):
    #     net.eval()
    #     cur_val_loss = []
    #     x, y = DataSet.shuffle_unison(data.X_test, data.Y_img_test)
    #     for i in range(data.X_test.shape[0]):
    #         input_x = torch.tensor(np.float32(x[i:(i + 1)]), device=device)
    #         target_y = torch.tensor(np.float32(y[i:(i + 1)]), device=device)

    #         optimizer.zero_grad()  # zero the gradient buffers
    #         output_y = net(input_x)
    #         loss = criterion(output_y, target_y)
    #         cur_val_loss = np.append(cur_val_loss, [loss])

    #     return np.mean(cur_val_loss)

class DataLoaderWrapper():
    def __init__(self, full_path, batch_size=200, type='rng'):
        with open(full_path, 'rb') as f:
            data = pickle.load(f)

        self.train_data, self.test_data = self.shuffle_split_dev(data, 0.85, type)

        if type=='rng':
            self.train_set = TensorDataset(self.train_data[0], self.normalize(self.train_data[1]))
            self.test_set = TensorDataset(self.test_data[0], self.normalize(self.test_data[1]))
        else:
            self.train_set = TensorDataset(self.train_data[0], self.train_data[1])
            self.test_set = TensorDataset(self.test_data[0], self.test_data[1])

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)

    def get_max_min(self, ranges):
        return ranges.max().item(), ranges.min().item()

    # shuffle dataset and split into train and test set
    def shuffle_split_dev(self, data, ratio, type):
        assert data[0].shape[0] == data[1].shape[0], "Dataset tensors are not of equal length."
        
        ran_ids = torch.randperm(data[0].shape[0])
        ratio_id = int(data[0].shape[0] * ratio)

        pose, rng, img = data
        x = pose
        if type=='rng':
            y = rng
        elif type=='img':
            y = img
        else:
            raise ValueError("Wrong type entered.")

        train = (
            x[ran_ids][ratio_id:],
            y[ran_ids][ratio_id:],
        )
        test = (
            x[ran_ids][:ratio_id],
            y[ran_ids][:ratio_id],
        )

        return train, test

    def normalize(self, ranges):
        ranges[ranges == float("Inf")] = 15
        ranges = (ranges - ranges.min()) / (ranges.max() - ranges.min())
        return ranges
