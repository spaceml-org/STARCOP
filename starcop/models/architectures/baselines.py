import torch.nn as nn
import torch 
from . import layer_factory


class SingleConv(nn.Module):
    """
    Single convolutional layer model
    """
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_class, 1)
        )

    def forward(self, x):
        
        res = self.conv(x)
        
        return res


class SimpleCNN(nn.Module):
    """
    5-layer fully conv CNN
    """
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.conv = nn.Sequential(
            layer_factory.double_conv(n_channels, 64),
            layer_factory.double_conv(64, 128),
            nn.Conv2d(128, n_class, 1)
        )
        
    def forward(self, x):
        
        res = self.conv(x)
        
        return res
    
class SimpleCNN_v2(nn.Module):
    def __init__(self,input_channel_size=13, output_channel_size=12):
        super().__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=input_channel_size,
                                  out_channels=output_channel_size,
                                  kernel_size=1,
                                  stride=1))
    def forward(self, x):
        x = self.cnn_layers(x)
        return x

    
class SimpleCNN_v3(nn.Module):
    def __init__(self,input_channel_size=13, output_channel_size=12):
        super().__init__()
        self.cnn_layers=torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=input_channel_size,
                                  out_channels=input_channel_size,
                                  kernel_size=1,
                                  stride=1),
                        torch.nn.Conv2d(in_channels=input_channel_size,
                                  out_channels=output_channel_size,
                                  kernel_size=1,
                                  stride=1))
    def forward(self, x):
        x = self.cnn_layers(x)
        return x
