import torch

class EncoderTCNBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=True):
    super().__init__()
    self.conv = torch.nn.Conv1d(
        in_channels, 
        out_channels, 
        kernel_size, 
        dilation=dilation, 
        padding=0,
        bias=True)
    # Activation function
    if activation:
      self.act = torch.nn.PReLU()
    self.kernel_size = kernel_size
    self.dilation = dilation

  def forward(self, x):
    x = self.conv(x)
    if hasattr(self, "act"):
      x = self.act(x)
    return x

class EncoderTCN(torch.nn.Module):
  def __init__(self, n_inputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False):
    super().__init__()
    self.kernel_size = kernel_size
    self.n_channels = n_channels
    self.dilation_growth = dilation_growth
    self.n_blocks = n_blocks
    self.use_kl = use_kl
    self.latent_dim = latent_dim

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
        self.blocks.append(EncoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation=act))
        print(f"Appended block {n} with in_ch={in_ch}, kernel_size={kernel_size}, out_ch={out_ch}, dilation={dilation}.")
        in_ch = out_ch  # Update in_ch for the next block

    # Use 1D convolutions to compute mean and log-variance
    self.conv_mu = torch.nn.Conv1d(in_ch, latent_dim, 1)
    self.conv_logvar = torch.nn.Conv1d(in_ch, latent_dim, 1)

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