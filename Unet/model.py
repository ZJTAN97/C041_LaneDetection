import torch
import torch.nn as nn

"""
https://www.youtube.com/watch?v=67r38S7Y-mA
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x= self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x= self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels+out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        """ Encoder """
        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.bottleneck = ConvBlock(512, 1024)

        """ Decoder """
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0) # binary
        self.sigmoidify = nn.Sigmoid()
    
    def forward(self, inputs):
        skip1, pool1 = self.encoder1(inputs)
        skip2, pool2 = self.encoder2(pool1)
        skip3, pool3 = self.encoder3(pool2)
        skip4, pool4 = self.encoder4(pool3)

        bottleneck = self.bottleneck(pool4)

        decoded1 = self.decoder1(bottleneck, skip4)
        decoded2 = self.decoder2(decoded1, skip3)
        decoded3 = self.decoder3(decoded2, skip2)
        decoded4 = self.decoder4(decoded3, skip1)

        outputs = self.outputs(decoded4)
        final = self.sigmoidify(outputs)

        return final


if __name__ == "__main__":
    ## [batch, channels, height, width]
    print('you ran this file directly.')
    inputs = torch.randn((1, 3, 512, 512))
    print(inputs.shape)
    model = Unet()
    y = model(inputs)
    print(y.shape)
