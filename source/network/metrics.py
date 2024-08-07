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

def spectral_distance(x, y):
    scales = [2048, 1024, 512, 256, 128]
    x = multiscale_stft(x, scales, .75)
    y = multiscale_stft(y, scales, .75)

    lin = sum(list(map(lin_distance, x, y)))
    log = sum(list(map(log_distance, x, y)))

    return lin + log

def plot_spectrums_tensorboard(writer, x, y, scales=[2048, 1024, 512, 256, 128], step=0):
    x_stfts = multiscale_stft(x, scales, .75)
    y_stfts = multiscale_stft(y, scales, .75)
    
    for i, scale in enumerate(scales):
        writer.add_image(f'Input STFT (scale {scale})', x_stfts[i][0], step, dataformats='HW')
        writer.add_image(f'Output STFT (scale {scale})', y_stfts[i][0], step, dataformats='HW')

def plot_distance_spectrums_tensorboard(writer, x, y, scales=[2048, 1024, 512, 256, 128], step=0):
    x_stfts = multiscale_stft(x, scales, .75)
    y_stfts = multiscale_stft(y, scales, .75)
    
    for i, scale in enumerate(scales):
        lin_dist = (x_stfts[i] - y_stfts[i]).abs()[0]
        log_dist = (torch.log(x_stfts[i] + 1e-7) - torch.log(y_stfts[i] + 1e-7)).abs()[0]
        
        writer.add_image(f'Linear Distance (scale {scale})', lin_dist, step, dataformats='HW')
        writer.add_image(f'Log Distance (scale {scale})', log_dist, step, dataformats='HW')