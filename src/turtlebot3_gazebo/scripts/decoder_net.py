import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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
        x = x.view(-1, 8, 15, 20)
        x = self.conv_layers(x)
        return x

    @staticmethod
    def train_net_img(net, data, device, net_id, max_epochs=10, batch_size=200):
        print("Starting training using {} data.".format(data.cat))

        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)
        criterion = nn.MSELoss()

        epoch_loss = []
        # val_loss = []
        batch_loss = []

        for epoch in range(max_epochs):
            cur_batch_loss = []
            x, y = data.shuffle_unison(data.X_train, data.Y_img_train)
            net.train()
            for i  in range(data.X_train.shape[0] // batch_size):
                input_x = torch.tensor(np.float32(x[i * batch_size:(i + 1) * batch_size]), device=device)
                target_y = torch.tensor(np.float32(y[i * batch_size:(i + 1) * batch_size]), device=device)

                optimizer.zero_grad()  # zero the gradient buffers
                output_y = net(input_x)
                loss = criterion(output_y, target_y)
                loss.backward()
                optimizer.step()  # Does the update
                loss = loss.item()

                cur_batch_loss = np.append(cur_batch_loss, [loss])
                batch_loss = np.append(batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            # cur_val_loss = Net.eval_test_set(net, data, device, optimizer, criterion)
            # val_loss = np.append(val_loss, [np.mean(cur_val_loss)])
            
            if epoch%1==0:
                print('------ Epoch ', epoch, '--------LR:', scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                # print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(), ImageNet.SAVE_PATH + "img/" + data.cat + str(net_id) + ".pt")
        
        torch.save(net.state_dict(), ImageNet.SAVE_PATH + "img/" + data.cat + str(net_id) + ".pt")

        return epoch_loss, batch_loss

    @staticmethod
    def eval_test_set(net, data, device, optimizer, criterion):
        net.eval()
        cur_val_loss = []
        x, y = DataSet.shuffle_unison(data.X_test, data.Y_img_test)
        for i in range(data.X_test.shape[0]):
            input_x = torch.tensor(np.float32(x[i:(i + 1)]), device=device)
            target_y = torch.tensor(np.float32(y[i:(i + 1)]), device=device)

            optimizer.zero_grad()  # zero the gradient buffers
            output_y = net(input_x)
            loss = criterion(output_y, target_y)
            cur_val_loss = np.append(cur_val_loss, [loss])

        return np.mean(cur_val_loss)

class RangeNet(nn.Module):

    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_conv_nets/")

    def __init__(self):
        super(RangeNet, self).__init__()
        self.ff_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 90),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            # shape 1 * 90
            nn.ConvTranspose1d(1, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 1, stride=1, padding=0),
            nn.ReLU(),
            # shape 1 * 180
            nn.ConvTranspose1d(1, 1, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(1, 1, 1, stride=1, padding=0),
            nn.ReLU()
            # shape 1 * 360
        )


    def forward(self, x):
        x = self.ff_layers(x)
        x = x.view(-1, 1, 90)
        x = self.conv_layers(x)
        return x


    @staticmethod
    def train_net(net, data, device, net_id, max_epochs=100, batch_size=400):
        print("Starting training using {} data.".format(data.cat))

        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=2000, gamma=0.95)
        criterion = nn.MSELoss()

        epoch_loss = []
        # val_loss = []
        batch_loss = []

        for epoch in range(max_epochs):
            cur_batch_loss = []
            x, y = data.shuffle_unison(data.X_train, data.Y_rng_train)
            net.train()
            for i  in range(data.X_train.shape[0] // batch_size):
                input_x = torch.tensor(np.float32(x[i * batch_size:(i + 1) * batch_size]), device=device)
                target_y = torch.tensor(np.float32(y[i * batch_size:(i + 1) * batch_size]), device=device)

                optimizer.zero_grad()  # zero the gradient buffers
                output_y = net(input_x)
                loss = criterion(output_y, target_y)
                loss.backward()
                optimizer.step()  # Does the update
                loss = loss.item()
                
                cur_batch_loss = np.append(cur_batch_loss, [loss])
                batch_loss = np.append(batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            # cur_val_loss = Net.eval_test_set(net, data, device, optimizer, criterion)
            # val_loss = np.append(val_loss, [np.mean(cur_val_loss)])
            
            if epoch%10==0:
                print('------ Epoch ', epoch, '--------LR:', scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                # print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(), RangeNet.SAVE_PATH + "rng/" + data.cat + str(net_id) + ".pt")
        
        torch.save(net.state_dict(), RangeNet.SAVE_PATH + "rng/" + data.cat + str(net_id) + ".pt")

        return epoch_loss, batch_loss

    @staticmethod
    def eval_test_set(net, data, device, optimizer, criterion):
        net.eval()
        cur_val_loss = []
        x, y = DataSet.shuffle_unison(data.X_test, data.Y_img_test)
        for i in range(data.X_test.shape[0]):
            input_x = torch.tensor(np.float32(x[i:(i + 1)]), device=device)
            target_y = torch.tensor(np.float32(y[i:(i + 1)]), device=device)

            optimizer.zero_grad()  # zero the gradient buffers
            output_y = net(input_x)
            loss = criterion(output_y, target_y)
            cur_val_loss = np.append(cur_val_loss, [loss])

        return np.mean(cur_val_loss)

class DataSet:

    LOAD_PATH = os.path.join(os.path.dirname(__file__), "data/")

    def __init__(self, cat):
        self.cat = cat

        if self.cat == 'warehouse':
            with open(DataSet.LOAD_PATH + "data_warehouse.pkl", 'rb') as f:
                self.data_set = pickle.load(f)

        elif self.cat == 'office':
            with open(DataSet.LOAD_PATH + "data_office.pkl", 'rb') as f:
                self.data_set = pickle.load(f)
        
        else:
            raise ValueError('Please enter a valid category.')
        
        self.pose = self.data_set[0]
        self.range = self.data_set[1]
        self.image = self.data_set[2]

        self.t = torch.tensor([self.pose, self.range])

        # self.X_train = self.train_set[0]
        # self.Y_rng_train = self.rm_inf(self.train_set[1])
        # self.Y_img_train = self.train_set[2]

        # self.Y_img_train = self.Y_img_train.reshape([-1, 1, self.Y_img_train.shape[1], self.Y_img_train.shape[2]])
        # self.Y_rng_train = self.Y_rng_train.unsqueeze(1)
        # # self.Y_rng_train = self.Y_rng_train.reshape([-1, 1, self.Y_rng_train.shape[1]])

        # self.X_test = self.test_set[0]
        # self.Y_rng_test = self.rm_inf(self.test_set[1])
        # self.Y_img_test = self.test_set[2]

        self.length = self.data_set[0].shape[0]

    @staticmethod
    def shuffle_unison(a, b):
        assert a.shape[0] == len(b)
        p = np.random.permutation(len(b))
        return a[p], b[p]

    def rm_inf(self, a):
        a[a == float("Inf")] = 15
        return a

if __name__ == "__main__":
    data_set = DataSet('warehouse')
    print(data_set.t)