import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.dropout import Dropout2d

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()        
        self.squeeze = nn.Sequential(
            nn.Conv2d(inplanes, squeeze_planes, kernel_size=1),
            nn.BatchNorm2d(squeeze_planes),
            nn.LeakyReLU(inplace=True)
        )

        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1),
            nn.BatchNorm2d(expand1x1_planes),
            nn.LeakyReLU(inplace=True)
        )

        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand3x3_planes),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)

        return x

class SqueezePoseNetEncoder(nn.Module):

    def __init__(self, dropout=0.5):
        super(SqueezePoseNetEncoder, self).__init__()
    
        self.squeeze = nn.Sequential(
            # conv1
            nn.Conv2d(1, 96, kernel_size=7, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(inplace=True),
            # pool1
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # fire2 & concat
            Fire(96, 16, 64, 64),
            # fire3 & concat
            Fire(128, 16, 64, 64),
            # fire4 & concat
            Fire(128, 32, 128, 128),
            # pool4
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # frie5 & concat
            Fire(256, 32, 128, 128),
            # fire6 & concat
            Fire(256, 48, 192, 192),
            # fire7 & concat
            Fire(384, 48, 192, 192),
            # fire8 & concat
            Fire(384, 64, 256, 256),
            # pool8
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # fire9 & concat
            Fire(512, 64, 256, 256),
            # drop9
            nn.Dropout(p=dropout), 
            # conv_final
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.BatchNorm2d(1000),
            nn.LeakyReLU(inplace=True),
            # # pool10
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.pose = nn.Sequential(
            # fc11
            nn.Linear(1000, 500),
            nn.LeakyReLU(inplace=True),
            # position x
            nn.Linear(500, 2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        print(x.shape)
        x = x.view(-1, 1, 1000)
        x = self.pose(x)

        return x

class FireDec(nn.Module):

    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super(FireDec, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(expand1x1_planes + expand3x3_planes, squeeze_planes, kernel_size=1),
            nn.BatchNorm2d(squeeze_planes),
            nn.LeakyReLU(inplace=True)
        )

        self.expand1x1 = nn.Sequential(
            nn.Conv2d(inplanes, expand1x1_planes, kernel_size=1),
            nn.BatchNorm2d(expand1x1_planes),
            nn.LeakyReLU(inplace=True)
        )

        self.expand3x3 = nn.Sequential(
            nn.Conv2d(inplanes, expand3x3_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand3x3_planes),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)
        x = self.squeeze(x)

        return x

class SqueezePoseNetDecoder(nn.Module):

    def __init__(self):
        super(SqueezePoseNetDecoder, self).__init__()

        self.pose_dec = nn.Sequential(
            # position x
            nn.Linear(2, 500),
            nn.LeakyReLU(inplace=True),
            # fc11
            nn.Linear(500, 1000),
            nn.LeakyReLU(inplace=True)
        )

        self.squeeze_dec = nn.Sequential(
            # pool10
            nn.ConvTranspose2d(1000, 1000, kernel_size=14, stride=1, padding=0),
            # conv_final
            nn.Conv2d(1000, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            # fire9 & concat
            FireDec(512, 512, 256, 256),
            # pool8
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2),
            # fire8 & concat
            FireDec(512, 384, 256, 256),
            # fire7 & concat
            FireDec(384, 384, 192, 192),
            # fire6 & concat
            FireDec(384, 256, 192, 192),
            # fire5 & concat
            FireDec(256, 256, 128, 128),
            # pool4
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),
            # fire4 & concat
            FireDec(256, 128, 128, 128),
            # fire3 & concat
            FireDec(128, 128, 64, 64),
            # fire2 & concat
            FireDec(128, 96, 64, 64),
            # pool1
            nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2),
            # conv1
            nn.ConvTranspose2d(96, 1, kernel_size=10, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pose_dec(x)
        x = x.view(-1, 1000, 1, 1)
        x = self.squeeze_dec(x)
        return x

class SqueezePoseNet(nn.Module):

    def __init__(self, channels, classes):
        super(SqueezePoseNet, self).__init__()

        self.encoder = SqueezePoseNetEncoder(channels, classes)
        self.decoder = SqueezePoseNetDecoder(channels, classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    net = SqueezePoseNetDecoder()

    img = torch.rand((1, 1, 2))

    out = net.forward(img)

    print(out.shape)