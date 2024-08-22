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
    signal = signal.view(b * c, t)
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
