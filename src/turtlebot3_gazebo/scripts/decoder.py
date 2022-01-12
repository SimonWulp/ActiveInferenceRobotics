import torch
from torch import nn

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.ff_layers=nn.Sequential( 
            nn.Linear(3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 8 * 128),
            nn.ReLU(),
        )

        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),

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
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ff_layers(x)        
        x = x.view(-1, 128, 8, 8)
        x = self.conv_layers(x)
        return x

if __name__ == "__main__":
    net = ConvDecoder()

    pose = torch.rand((1, 1, 2))

    out = net.forward(pose)

    print(out.shape)