import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as wn

class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation, activation='prelu'):
        super().__init__()
        self.dilated_conv = wn(nn.Conv1d(channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation))
        self.conv_1x1 = wn(nn.Conv1d(channels, channels, 1))
        self.activation = nn.PReLU() if activation == 'prelu' else nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.dilated_conv(self.activation(x))
        return x + self.conv_1x1(self.activation(y))

class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size, num_layers=3, activation='prelu'):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualLayer(channels, kernel_size, dilation=3**i, activation=activation)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation='prelu', use_skip=True, use_wn=True, use_residual=True, stride=1, padding=0):
        super().__init__()
        conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=stride//2 + padding, stride=stride, bias=True, output_padding=stride//2)
        self.conv = wn(conv) if use_wn else conv
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.residual_stack = ResidualStack(out_channels, kernel_size=3, activation=activation) if use_residual else None
        self.activation = nn.PReLU() if activation == 'prelu' else nn.LeakyReLU(0.2)

        gate = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.gate = wn(gate) if use_wn else gate
        self.sigmoid = nn.Sigmoid()

        self.use_skip = use_skip

    def forward(self, x, skip=None):
        x = self.activation(self.conv(x))
        
        if self.residual_stack:
            x = self.residual_stack(x)

        if self.use_skip and skip is not None:
            gate = self.sigmoid(self.gate(torch.cat([x, skip], dim=1)))
            x = x + gate * skip
        return x

class NoiseGenerator(nn.Module):
    def __init__(self, n_channels, noise_bands=4):
        super().__init__()
        layers = []
        current_channels = n_channels
        for i in range(3):
            out_channels = max(current_channels // 2, noise_bands)
            layers.append(wn(nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1)))
            if i < 2:
                layers.append(nn.LeakyReLU(0.2))
            current_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        amp = torch.sigmoid(self.net(x) - 5)
        noise = torch.randn_like(amp) * amp
        return noise.sum(dim=1, keepdim=True)

class DecoderTCN(nn.Module):
    def __init__(self, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False, use_skip=True, use_noise=True, noise_bands=4, use_wn=True, use_residual=True, dilate_conv=False, use_latent=False, activation='prelu', stride=1, padding=0):    
        super().__init__()
        self.use_kl = use_kl
        self.use_skip = use_skip
        self.use_noise = use_noise
        self.use_latent = use_latent
        self.latent_dim = latent_dim

        initial_channels = n_channels * (2 ** (n_blocks - 1))
        self.initial_channels = initial_channels

        if use_latent == 'dense':
            self.dense_expand = wn(nn.Linear(latent_dim, initial_channels)) if use_wn else nn.Linear(latent_dim, initial_channels)
        elif use_kl or use_latent == 'conv':
            conv = nn.ConvTranspose1d(latent_dim, initial_channels, kernel_size, dilation=dilation_growth**n_blocks if dilate_conv else 1)
            self.conv_decode = wn(conv) if use_wn else conv
            
        self.blocks = nn.ModuleList()

        in_ch = initial_channels
        for n in range(n_blocks):
            out_ch = n_outputs if n == n_blocks - 1 else in_ch // 2
            dilation = dilation_growth ** (n_blocks - n)
            self.blocks.append(DecoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation=activation, use_skip=use_skip and n < n_blocks - 1, use_wn=use_wn, use_residual=use_residual, stride=stride, padding=padding))
            in_ch = out_ch

        if use_noise:
            self.noise_generator = NoiseGenerator(n_outputs, noise_bands=noise_bands)

    def forward(self, x, skips):
        if self.use_latent == 'conv' or self.use_kl:
            x = self.conv_decode(x)
        elif self.use_latent == 'dense':
            batch_size, latent_dim, time_steps = x.shape
            x = self.dense_expand(x).view(batch_size, time_steps, self.initial_channels).transpose(1, 2)

        for i, (block, skip) in enumerate(zip(self.blocks, skips)):
            x = block(x, skip)

        if self.use_noise:
            noise = self.noise_generator(x)
            x = x + noise

        return x