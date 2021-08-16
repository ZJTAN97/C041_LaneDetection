import torch
from torch.nn.modules.conv import Conv2d
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
import torch.nn.functional as F


class ImageSegmentationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, labels)   # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)                  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



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


class Unet(ImageSegmentationBase):
    def __init__(self):
        super().__init__()
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

        return outputs


if __name__ == "__main__":
    ## [batch, channels, height, width]
    inputs = torch.randn((2, 3, 512, 512))
    model = Unet()
    y = model(inputs)
    print(y.shape)
