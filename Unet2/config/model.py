from . import config
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        # conv -> ReLu -> conv
        return self.conv2(self.relu(self.conv1(x)))
    

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store encoder blocks and maxpooling layer
        self.encoder_blocks = ModuleList( [ Block(channels[i], channels[i+1]) for i in range(len(channels) - 1) ] )
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # empty list to store intermediate outputs
        block_outputs = []
        
        for block in self.encoder_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.decoder_blocks = ModuleList( [ ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1) ] )

    
    def crop(self, encoding_features, x):
        # grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
        (_, _, H, W) = x.shape
        encoding_features = CenterCrop([H, W])(encoding_features)

        return encoding_features

    
    def forward(self, x, encoding_features):
        for i in range(len(self.channels)-1):
            x = self.decoder_blocks[i](x)

			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
            encoding_feature = self.crop(encoding_features[i], x)
            x = torch.cat([x, encoding_feature], dim=1)
            x = self.decoder_blocks[i](x)
        
        return x


class UNet(Module):
    def __init__(self, encoding_channels=(3, 16, 32, 64), decoding_channels=(64, 32, 16), num_classes=1, retain_dim=True, output_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        self.encoder = Encoder(encoding_channels)
        self.decoder = Decoder(decoding_channels)

        self.head = Conv2d(decoding_channels[-1], num_classes, 1)
        self.retain_dim = retain_dim
        self.output_size = output_size

    def forward(self, x):
        encoding_features = self.encoder(x)
        decoding_features = self.decoder(encoding_features[::-1][0], encoding_features[::-1][1:])
        map = self.head(decoding_features)

        if self.retain_dim:
            map = F.interpolate(map, self.output_size)
        
        return map
