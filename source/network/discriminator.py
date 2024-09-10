import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, n_layers=5):
        super().__init__()
        
        layers = []
        in_ch = in_channels
        out_ch = base_channels
        
        for i in range(n_layers):
            layers.append(DiscriminatorBlock(in_ch, out_ch, kernel_size=15, stride=2, padding=7))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        
        self.main = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            features.append(x)
        
        output = self.final_conv(x)
        return output, features