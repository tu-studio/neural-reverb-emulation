import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn

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
    def __init__(self, n_inputs=1, n_channels=64, n_layers=5, kernel_size=15, padding=7, stride=2):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        in_ch = n_inputs
        out_ch = n_channels

        print(f"Building Discriminator with {n_layers} layers")
        for i in range(n_layers):
            self.blocks.append(DiscriminatorBlock(in_ch, out_ch, kernel_size, stride, padding))
            print(f"Appended layer {i} with in_ch={in_ch}, out_ch={out_ch}, kernel_size={kernel_size}, stride={stride}, padding={padding}")
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)

        self.final_conv = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        output = self.final_conv(x)
        return output, features