import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        
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
        return torch.cat(
            [self.expand1x1(x), self.expand3x3(x)], 1
        )

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
            # pool10
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.pose = nn.Sequential(
            # fc11
            nn.Linear(1000, 500),
            nn.LeakyReLU(),
            # position x
            nn.Linear(500, 2),
            nn.LeakyReLU()
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = x.view(-1, 1, 1000)
        x = self.pose(x)

        return x

class FireDec(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireDec, self).__init__()
        self.expand1x1 = nn.Conv2d(inplanes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(inplanes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv2d(expand1x1_planes + expand3x3_planes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.expand1x1_activation(self.expand1x1(x))
        z = self.expand3x3_activation(self.expand3x3(x))
        x = torch.cat((y,z), 1)
        x = self.squeeze_activation(self.squeeze(x))
        return x

class SqueezePoseNetDecoder(nn.Module):

    def __init__(self, out_channels, in_classes):
        super(SqueezePoseNetDecoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_classes, 3 * 14 * 14),
            nn.ReLU()
        )

        self.inverse_classifier_conv = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.inverse_feature_block4 = nn.Sequential(
            FireDec(512, 512, 256, 256),
        )

        self.inverse_feature_block3 = nn.Sequential(
            FireDec(512, 384, 256, 256),
            FireDec(384, 384, 192, 192),
            FireDec(384, 256, 192, 192),
            FireDec(256, 256, 128, 128),
        )

        self.inverse_feature_block2 = nn.Sequential(
            FireDec(256, 128, 128, 128),
            FireDec(128, 128, 64, 64),
            FireDec(128, 96, 64, 64),
        )

        self.inverse_feature_block1 = nn.Sequential(
            nn.ConvTranspose2d(96, out_channels, kernel_size=10, stride=2, padding=1),
        )

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2)


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 3, 14, 14)
        x = self.inverse_classifier_conv(x)
        x = self.inverse_feature_block4(x)
        x = self.deconv1(x)
        x = self.inverse_feature_block3(x)
        x = self.deconv2(x)
        x = self.inverse_feature_block2(x)
        x = self.deconv3(x)
        x = self.inverse_feature_block1(x)
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
    net = SqueezePoseNetEncoder()

    img = torch.rand((1, 1, 256, 256))

    out = net.forward(img)

    print(out.shape)