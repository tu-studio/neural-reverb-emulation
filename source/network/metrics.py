import torch

def multiscale_stft(signal, scales, overlap):
    """
    Compute a stft on several scales, with a constant overlap value.
    Parameters
    ----------
    signal: torch.Tensor
        input signal to process ( B X C X T )
    
    scales: list
        scales to use
    overlap: float
        overlap between windows ( 0 - 1 )
    """
    b, c, t = signal.shape
    signal = signal.contiguous().reshape(b * c, t)
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts

def lin_distance(x, y):
    return torch.norm(x - y) / torch.norm(x)

def log_distance(x, y):
    return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

def spectral_distance(x, y, scales = [2048, 1024, 512, 256, 128]):
    x = multiscale_stft(x, scales, .75)
    y = multiscale_stft(y, scales, .75)

    lin = sum(list(map(lin_distance, x, y)))
    log = sum(list(map(log_distance, x, y)))

    return lin + log

def single_stft_loss(x, y, scale=2048, overlap=0.75):
    """
    Compute loss using a single STFT at scale 2048 and linear distance.
    """
    x_stft = multiscale_stft(x, [scale], overlap)[0]
    y_stft = multiscale_stft(y, [scale], overlap)[0]
    return lin_distance(x_stft, y_stft)

def fft_loss(x, y):
    """
    Compute loss using a single FFT and linear distance.
    """
    x_fft = torch.fft.fft(x)
    y_fft = torch.fft.fft(y)
    
    return lin_distance(x_fft.abs(), y_fft.abs())


class CombinedLoss(torch.nn.Module):
    def __init__(self, mse_weight=1.0, spectral_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        spec_dist = spectral_distance(output, target)
        return self.mse_weight * mse + self.spectral_weight * spec_dist
