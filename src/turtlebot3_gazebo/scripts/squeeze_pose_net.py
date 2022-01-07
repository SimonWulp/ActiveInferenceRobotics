import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        y = self.expand1x1_activation(self.expand1x1(x))
        z = self.expand3x3_activation(self.expand3x3(x))
        x = torch.cat((y,z), 1)
        return x

class SqueezeSegNetEncoder(nn.Module):

    def __init__(self, in_channels):
        super(SqueezeSegNetEncoder, self).__init__()
    
        self.feature_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
        )

        self.feature_block2 = nn.Sequential(
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
        )
        self.feature_block3 = nn.Sequential(
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
        )
        self.feature_block4 = nn.Sequential(
            Fire(512, 64, 256, 256),
        )

        self.classifier_conv = nn.Sequential(
            nn.Conv2d(512, 1000, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(96, 96, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.feature_block1(x)
        x = self.conv1(x)
        x = self.feature_block2(x)
        x = self.conv2(x)
        x = self.feature_block3(x)
        x = self.conv3(x)
        x = self.feature_block4(x)
        x = self.classifier_conv(x)

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

class SqueezeSegNetDecoder(nn.Module):

    def __init__(self, out_classes):
        super(SqueezeSegNetDecoder, self).__init__()

        self.inverse_classifier_conv = nn.Sequential(
            nn.Conv2d(1000, 512, kernel_size=1),
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
            nn.ConvTranspose2d(96, out_classes, kernel_size=10, stride=2, padding=1),
        )

        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, 2)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, 2)
        self.deconv3 = nn.ConvTranspose2d(96, 96, 3, 2)


    def forward(self, x):
        x = self.inverse_classifier_conv(x)
        x = self.inverse_feature_block4(x)
        x = self.deconv1(x)
        x = self.inverse_feature_block3(x)
        x = self.deconv2(x)
        x = self.inverse_feature_block2(x)
        x = self.deconv3(x)
        x = self.inverse_feature_block1(x)
        return x

class SqueezeSegNet(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(SqueezeSegNet, self).__init__()

        self.encoder = SqueezeSegNetEncoder(in_channels)
        self.decoder = SqueezeSegNetDecoder(out_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    net = SqueezeSegNet(3, 3)

    img_in = torch.rand((1, 3, 244, 244))

    latent = net.encoder.forward(img_in)

    img_out = net.decoder.forward(latent)

    print(latent.shape)
    print(img_out.shape)