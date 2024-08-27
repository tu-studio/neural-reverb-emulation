import torch

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
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_skip = use_skip

    def forward(self, x, skip=None):
        x = self.conv(x)
        if hasattr(self, "act"):
            x = self.act(x)
        if self.use_skip:
            x = x + self.alpha * skip
        return x

class DecoderTCN(torch.nn.Module):
    def __init__(self, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4, latent_dim=16, use_kl=False, use_skip=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dilation_growth = dilation_growth
        self.n_blocks = n_blocks
        self.use_kl = use_kl
        self.use_skip = use_skip

        # Add a convolutional layer to leave latent space
        initial_channels = n_channels * (2 ** (n_blocks - 1))
        self.conv_decode = torch.nn.Conv1d(latent_dim, initial_channels, 1)

        print(f"Building DecoderTCN with {n_blocks} blocks")
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
            print(f"Appended block {n} with in_ch={in_ch}, kernel_size={kernel_size}, out_ch={out_ch}, dilation={dilation}.")
            self.blocks.append(DecoderTCNBlock(in_ch, out_ch, kernel_size, dilation, activation=act, use_skip=use_skip))
            if (n+1) != n_blocks:
                in_ch = out_ch # Update in_ch for the next block

    def forward(self, x, skips):
        if self.use_kl:
            x = self.conv_decode(x)

        for i, (block, skip) in enumerate(zip(self.blocks, skips)):
            x = block(x, skip)
        return x
    
    def get_alpha_values(self):
        return [block.alpha.item() for block in self.blocks]