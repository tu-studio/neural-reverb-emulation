import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            dilation=dilation, 
            padding=0,
            bias=True)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        if activation:
            self.act = nn.PReLU()
        self.res = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        nn.init.xavier_uniform_(self.res.weight)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        x_in = x
        x = self.conv(x)
        if hasattr(self, "act"):
            x = self.act(x)
        x_res = self.res(x_in)
        x_res = x_res[..., (self.kernel_size-1)*self.dilation:]
        x = x + x_res
        return x

class TCN(nn.Module):
    def __init__(self, n_inputs=1, n_outputs=1, n_blocks=10, kernel_size=13, n_channels=64, dilation_growth=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dilation_growth = dilation_growth
        self.n_blocks = n_blocks
        self.stack_size = n_blocks

        self.blocks = nn.ModuleList()
        for n in range(n_blocks):
            if n == 0:
                in_ch = n_inputs
                out_ch = n_channels
            elif (n+1) == n_blocks:
                in_ch = n_channels
                out_ch = n_outputs
            else:
                in_ch = n_channels
                out_ch = n_channels
            
            dilation = dilation_growth ** n
            self.blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, activation=True))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    def compute_receptive_field(self):
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** (n % self.stack_size)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf