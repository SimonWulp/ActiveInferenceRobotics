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
from decoder_net import Net, OfficeDatasetImages, WarehouseDatasetImages

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

net = Net()
net.to(device)

data = OfficeDatasetImages()

def train_net(max_epochs=100, batch_size=200):
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.95)
    criterion = nn.MSELoss()

    epoch_train_loss = []
    for epoch in range(max_epochs):
        x, y = data.X_train, data.Y_train
        net.train()
        batch_train_loss = 0
        for i  in range(int(data.length / batch_size)):
            input_x = torch.from_numpy(np.float32(x[i * batch_size:(i + 1) * batch_size]))
            target_y = torch.from_numpy(np.float32(y[i * batch_size:(i + 1) * batch_size]))
            input_x, target_y = input_x.to(device), target_y.to(device)
            optimizer.zero_grad()  # zero the gradient buffers
            output_y = net(input_x)
            loss = criterion(output_y, target_y)
            loss.backward()
            optimizer.step()  # Does the update
            batch_train_loss += loss.item()
        scheduler.step()
        epoch_train_loss.append(batch_train_loss / batch_size)
        if epoch%1==0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, batch_train_loss / int(data.length) / batch_size))
    return epoch_train_loss

epoch_train_loss = train_net(5)

print(epoch_train_loss)

# Save the network after training has been finished
torch.save(net.state_dict(), 'net_end_of_training_office')