import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.weight_norm as wn

class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation, activation='prelu'):
        super().__init__()
        self.dilated_conv = wn(nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        ))
        self.conv_1x1 = wn(nn.Conv1d(channels, channels, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.prelu = nn.PReLU()
        self.activation = activation
    

    def forward(self, x):
        y = self.dilated_conv(self.activate(x))
        y = self.conv_1x1(self.activate(y))
        return x + y

    def activate(self, x):
        if self.activation == 'leaky_relu':
            return self.leaky_relu(x)
        elif self.activation == 'prelu':
            return self.prelu(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size, num_layers=3, activation='prelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualLayer(channels, kernel_size, dilation=3**i, activation='prelu')
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_skip=True, use_wn=True, use_residual=True, activation='prelu', stride=1):
        super().__init__()
        if use_wn:
            self.conv = wn(torch.nn.ConvTranspose1d(
                in_channels, 
                out_channels, 
                kernel_size, 
                dilation=dilation, 
                padding=0,
                bias=True,
                stride=stride))
        else:
            self.conv = torch.nn.ConvTranspose1d(
                in_channels, 
                out_channels, 
                kernel_size, 
                dilation=dilation, 
                padding=0,
                bias=True,
                stride=stride)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

        self.residual_stack = ResidualStack(out_channels, kernel_size=3, activation=activation)

        self.activation = activation
        self.prelu = torch.nn.PReLU()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        if use_wn:
            self.gate = wn(torch.nn.Conv1d(out_channels + out_channels, out_channels, 1))
        else:
            self.gate = torch.nn.Conv1d(out_channels + out_channels, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_skip = use_skip
        self.use_residual = use_residual

    def forward(self, x, skip=None):
        x = self.conv(x)
        if self.activation == 'leaky_relu':
            x = self.leaky_relu(x)
        elif self.activation == 'prelu':
            x = self.prelu(x)
        
        # print(x.shape)

        if self.use_residual:
            x = self.residual_stack(x)

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
                wn(nn.Conv1d(
                    current_channels,
                    max(current_channels // 2, noise_bands),
                    kernel_size=3,
                    padding=1
                ))
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
    def __init__(self, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False, use_skip=True, use_noise=True, noise_ratios=[4], noise_bands=4, use_wn=True, use_residual=True, activation='prelu', stride=1, dilate_conv=False):    
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
        if use_kl:
            if use_wn:
                self.conv_decode = wn(torch.nn.Conv1d(latent_dim, initial_channels, 1))
            else:
                self.conv_decode = torch.nn.Conv1d(latent_dim, initial_channels, 1)
        else:
            if use_wn:
                if dilate_conv:
                    self.conv_decode = wn(torch.nn.ConvTranspose1d(2 * latent_dim, initial_channels, kernel_size, padding=2, groups=2, dilation=dilation_growth**n_blocks))
                else:
                    self.conv_decode = wn(torch.nn.ConvTranspose1d(2 * latent_dim, initial_channels, kernel_size, padding=2, groups=2))
            else:
                if dilate_conv:
                    self.conv_decode = torch.nn.ConvTranspose1d(2 * latent_dim, initial_channels, kernel_size, padding=2, groups=2, dilation=dilation_growth**n_blocks)
                else:
                    self.conv_decode = torch.nn.ConvTranspose1d(2 * latent_dim, initial_channels, kernel_size, padding=2, groups=2)
            
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
                self.blocks.append(DecoderTCNBlock(in_ch, out_ch, kernel_size, dilation, use_skip=use_skip, use_wn=use_wn, use_residual=use_residual, activation=activation, stride=stride))
            else: 
                self.blocks.append(DecoderTCNBlock(in_ch, out_ch, kernel_size, dilation, use_skip=False, use_wn=use_wn, use_residual=use_residual, activation=activation, stride=stride))
            if (n+1) != n_blocks:
                in_ch = out_ch # Update in_ch for the next block

        if use_noise:
            self.noise_generator = NoiseGenerator(n_outputs, noise_bands=noise_bands)

    def forward(self, x, skips):
        
        x = self.conv_decode(x)

        for i, (block, skip) in enumerate(zip(self.blocks, skips)):
            x = block(x, skip)

        if self.use_noise:
            noise = self.noise_generator(x)
            x = x + noise

        return torch.tanh(x)
    