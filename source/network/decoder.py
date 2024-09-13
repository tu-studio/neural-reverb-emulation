import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DecoderTCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=True, use_skip=True):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation, 
            padding=0,
            bias=True)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

        # Activation function
        if activation:
            self.act = torch.nn.PReLU()

        # Residual connection
        # self.res = torch.nn.Conv1d(in_channels, out_channels, 1, bias=True)
        # torch.nn.init.xavier_uniform_(self.res.weight)

        # Learnable parameter for scaling the skip connection
        self.gate = torch.nn.Conv1d(in_channels + out_channels, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_skip = use_skip

    def forward(self, x, skip=None):
        x = self.conv(x)
        if hasattr(self, "act"):
            x = self.act(x)
        if self.use_skip:
            gate = self.sigmoid(self.gate(torch.cat([x, skip], dim=1)))
            x = x + gate * skip
        return x

class NoiseGenerator(nn.Module):
    def __init__(self, n_channels, noise_bands=4):
        super().__init__()
        self.n_channels = n_channels
        self.noise_bands = noise_bands

        # Create the network
        layers = []
        current_channels = n_channels
        for i in range(3):  # Reduce the number of steps
            layers.append(
                nn.Conv1d(
                    current_channels,
                    max(current_channels // 2, noise_bands),
                    kernel_size=3,
                    padding=1
                )
            )
            if i < 2:  # No activation on the last layer
                layers.append(nn.LeakyReLU(0.2))
            current_channels = max(current_channels // 2, noise_bands)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, n_channels, 1]
        amp = torch.sigmoid(self.net(x) - 5)  # Shape: [batch_size, noise_bands, length]
        noise = torch.randn_like(amp) * amp
        return noise.sum(dim=1, keepdim=True)

class DecoderTCN(nn.Module):
    def __init__(self, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False, use_skip=True, use_noise=True, noise_ratios=[4], noise_bands=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dilation_growth = dilation_growth
        self.n_blocks = n_blocks
        self.use_kl = use_kl
        self.use_skip = use_skip
        self.use_noise = use_noise

        # Add a convolutional layer to leave latent space
        initial_channels = n_channels * (2 ** (n_blocks - 1))
        self.conv_decode = torch.nn.Conv1d(latent_dim, initial_channels, 1)

        self.blocks = torch.nn.ModuleList()

        in_ch = n_channels * (2 ** (n_blocks - 1))
        for n in range(0, n_blocks):
            if n_blocks == 1:
                in_ch = n_channels
                out_ch = n_outputs
            elif (n+1) == n_blocks:
                in_ch = in_ch
                out_ch = n_outputs
            else:
                in_ch = in_ch
                out_ch = in_ch // 2 # Divide the number of channels at each block
            
            act = True
            dilation = dilation_growth ** (n_blocks - n)
            if (n+1) != n_blocks:
                self.blocks.append(DecoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation=act, use_skip=use_skip))
            else: 
                self.blocks.append(DecoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation=act, use_skip=False))
            if (n+1) != n_blocks:
                in_ch = out_ch # Update in_ch for the next block

        if use_noise:
            self.noise_generator = NoiseGenerator(n_outputs, noise_bands=noise_bands)

    def forward(self, x, skips):
        if self.use_kl:
            x = self.conv_decode(x)

        for i, (block, skip) in enumerate(zip(self.blocks, skips)):
            x = block(x, skip)

        if self.use_noise:
            noise = self.noise_generator(x)
            x = x + noise

        return x
    
    def get_alpha_values(self):
        return [block.alpha.item() for block in self.blocks]