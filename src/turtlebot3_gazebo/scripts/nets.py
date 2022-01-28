import torch
from torch import nn
import torch.nn.functional as F


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.ff_layers=nn.Sequential( 
            nn.Linear(2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5 * 5 * 128),
            nn.ReLU(),
        )

        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ff_layers(x)        
        x = x.view(-1, 128, 5, 5)
        x = self.conv_layers(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(18 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    net = Classifier()
    inp = torch.rand((1,3,80,80))

    outp = net.forward(inp)