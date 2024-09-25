import torch
import torch.nn.utils.weight_norm as wn

class EncoderTCNBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation, activation='prelu', use_wn=True, use_batch_norm=True):
    super().__init__()
    if use_wn:
      self.conv = wn(torch.nn.Conv1d(
          in_channels, 
          out_channels, 
          kernel_size, 
          dilation=dilation, 
          padding=0,
          bias=True))
    else: 
      self.conv = torch.nn.Conv1d(
          in_channels, 
          out_channels, 
          kernel_size, 
          dilation=dilation, 
          padding=0,
          bias=True)
          
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.activation = activation
    self.prelu = torch.nn.PReLU()
    self.leaky_relu = torch.nn.LeakyReLU(0.2)
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.use_batch_norm = use_batch_norm

  def forward(self, x):
    x = self.conv(x)
    if self.use_batch_norm:
      x = self.bn(x)
    if self.activation == 'leaky_relu':
      x = self.leaky_relu(x)
    elif self.activation == 'prelu':
      x = self.prelu(x)
    return x

class EncoderTCN(torch.nn.Module):
  def __init__(self, n_inputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False, use_wn=True, use_batch_norm=True, dilate_conv=False, use_latent=False, activation='prelu'):
    super().__init__()
    self.kernel_size = kernel_size
    self.n_channels = n_channels
    self.dilation_growth = dilation_growth
    self.n_blocks = n_blocks
    self.use_kl = use_kl
    self.latent_dim = latent_dim
    self.use_latent = use_latent

    self.blocks = torch.nn.ModuleList()
    print(f"Building EncoderTCN with {n_blocks} blocks")
    in_ch = n_inputs
    for n in range(n_blocks):
        if n == 0:
            out_ch = n_channels
            act = True
        else:
            out_ch = in_ch * 2  # Double the number of channels at each block
            act = True
        
        dilation = dilation_growth ** (n + 1)
        self.blocks.append(EncoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation=activation, use_wn=use_wn, use_batch_norm=use_batch_norm))
        print(f"Appended block {n} with in_ch={in_ch}, kernel_size={kernel_size}, out_ch={out_ch}, dilation={dilation}.")
        in_ch = out_ch  # Update in_ch for the next block

    if use_wn:
      self.conv_mu = wn(torch.nn.Conv1d(in_ch, latent_dim, 1))
      self.conv_logvar = wn(torch.nn.Conv1d(in_ch, latent_dim, 1))
      self.dense_latent = wn(torch.nn.Linear(in_ch, latent_dim))
      if dilate_conv:
        self.conv_latent = wn(torch.nn.Conv1d(in_ch, latent_dim, kernel_size, dilation=dilation))
      else:
        self.conv_latent = wn(torch.nn.Conv1d(in_ch, latent_dim, kernel_size))
    else:
      self.conv_mu = torch.nn.Conv1d(in_ch, latent_dim, 1)
      self.conv_logvar = torch.nn.Conv1d(in_ch, latent_dim, 1)
      self.dense_latent = torch.nn.Linear(in_ch, latent_dim)
      if dilate_conv:
        self.conv_latent = torch.nn.Conv1d(in_ch, latent_dim, kernel_size, dilation=dilation)    
      else:
        self.conv_latent = torch.nn.Conv1d(in_ch, latent_dim, kernel_size)

  def forward(self, x):
    encoder_outputs = [x]  # Include input as first element
    for block in self.blocks:
        x = block(x)
        encoder_outputs.append(x)
    
    if self.use_kl:
        # Compute mean and log-variance
        mu = torch.tanh(self.conv_mu(x))
        logvar = torch.nn.functional.softplus(self.conv_logvar(x))
        return mu, logvar, encoder_outputs
    else:
        if self.use_latent == 'conv':
            latent = self.conv_latent(x)
            encoder_outputs[-1] = latent 
        elif self.use_latent == 'dense':   
            batch_size, channels, time_steps = x.shape
            print(x.shape)
            x = x.transpose(1, 2).contiguous()  # [batch_size, time_steps, channels]
            x = x.view(-1, channels)  # [batch_size * time_steps, channels]
            latent = self.dense_latent(x)  # [batch_size * time_steps, latent_dim]
            latent = latent.view(batch_size, time_steps, self.latent_dim)  # [batch_size, time_steps, latent_dim]
            latent = latent.transpose(1, 2)  # [batch_size, latent_dim, time_steps]
            encoder_outputs[-1] = latent
            print(f"Latent shape: {latent.shape}")
        return encoder_outputs

  def reparameterize(self, mu, logvar):
      """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

  def compute_receptive_field(self):
      """Compute the receptive field in samples."""
      rf = self.kernel_size
      for n in range(1, self.n_blocks):
          dilation = self.dilation_growth ** (n % self.n_blocks)
          rf = rf + ((self.kernel_size - 1) * dilation)
      return rf