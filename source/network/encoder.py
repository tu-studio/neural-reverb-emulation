import torch
import torch.nn.utils.weight_norm as wn

class EncoderTCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation='prelu', use_wn=True, use_batch_norm=True, stride=1, padding=0):
        super().__init__()
        conv = torch.nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation, 
            padding=stride // 2 + padding,
            stride=stride,
            bias=True
        )
        self.conv = wn(conv) if use_wn else conv
        
        self.bn = torch.nn.BatchNorm1d(out_channels) if use_batch_norm else None
        self.activation = torch.nn.PReLU() if activation == 'prelu' else torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return self.activation(x)

class EncoderTCN(torch.nn.Module):
    def __init__(self, n_inputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False, use_wn=True, use_batch_norm=True, dilate_conv=False, use_latent=False, activation='prelu', stride=1, padding=0):
        super().__init__()
        self.use_kl = use_kl
        self.latent_dim = latent_dim
        self.use_latent = use_latent

        self.blocks = torch.nn.ModuleList()
        in_ch = n_inputs
        for n in range(n_blocks):
            out_ch = n_channels if n == 0 else in_ch * 2
            dilation = dilation_growth ** (n + 1)
            self.blocks.append(EncoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation, use_wn, use_batch_norm, stride, padding))
            in_ch = out_ch

        if use_kl:
            self.conv_mu = wn(torch.nn.Conv1d(in_ch, latent_dim, 1)) if use_wn else torch.nn.Conv1d(in_ch, latent_dim, 1)
            self.conv_logvar = wn(torch.nn.Conv1d(in_ch, latent_dim, 1)) if use_wn else torch.nn.Conv1d(in_ch, latent_dim, 1)
        elif use_latent == 'conv':
            conv = torch.nn.Conv1d(in_ch, latent_dim, kernel_size, dilation=dilation if dilate_conv else 1)
            self.conv_latent = wn(conv) if use_wn else conv
        elif use_latent == 'dense':
            self.dense_latent = wn(torch.nn.Linear(in_ch, latent_dim)) if use_wn else torch.nn.Linear(in_ch, latent_dim)

    def forward(self, x):
        encoder_outputs = [x]
        for block in self.blocks:
            x = block(x)
            encoder_outputs.append(x)
        
        if self.use_kl:
            mu = torch.tanh(self.conv_mu(x))
            logvar = torch.nn.functional.softplus(self.conv_logvar(x))
            return mu, logvar, encoder_outputs
        elif self.use_latent == 'conv':
            latent = self.conv_latent(x)
            encoder_outputs[-1] = latent
        elif self.use_latent == 'dense':
            latent = self.dense_latent(x.flatten(0, 1))
            encoder_outputs[-1] = latent
        return encoder_outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_receptive_field(self):
        rf = self.blocks[0].conv.kernel_size[0]
        for block in self.blocks[1:]:
            rf = rf + ((block.conv.kernel_size[0] - 1) * block.conv.dilation[0])
        return rf