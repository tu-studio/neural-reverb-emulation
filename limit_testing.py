import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from source.network.encoder import EncoderTCN
from source.network.decoder import DecoderTCN
from source.network.CombinedModels import CombinedEncoderDecoder
from source.utils import config
from source.network.ravepqmf import PQMF, center_pad_next_pow_2

# Load the parameters
params = config.Params('params.yaml')

# Load model parameters
n_bands = params["train"]["n_bands"]
latent_dim = params["train"]["latent_dim"]
kernel_size = params["train"]["kernel_size"]
n_blocks = params["train"]["n_blocks"]
dilation_growth = params["train"]["dilation_growth"]
n_channels = params["train"]["n_channels"]
use_kl = params["train"]["use_kl"]
use_skip = params["train"]["use_skip"]
use_noise = params["train"]["use_noise"]
use_wn = params["train"]["use_wn"]
use_batch_norm = params["train"]["use_batch_norm"]
use_residual = params["train"]["use_residual"]
use_latent = params["train"]["use_latent"]
dilate_conv = params["train"]["dilate_conv"]
activation = params["train"]["activation"]
stride = params["train"]["stride"]
padding = params["train"]["padding"]

# Initialize encoder and decoder
encoder = EncoderTCN(
    n_inputs=n_bands,
    kernel_size=kernel_size, 
    n_blocks=n_blocks, 
    dilation_growth=dilation_growth, 
    n_channels=n_channels,
    latent_dim=latent_dim,
    use_kl=use_kl,
    use_wn=use_wn,
    use_batch_norm=use_batch_norm,
    use_latent=use_latent,
    dilate_conv=dilate_conv,
    activation=activation,
    stride=stride,
    padding=padding
)

decoder = DecoderTCN(
    n_outputs=n_bands,
    kernel_size=kernel_size,
    n_blocks=n_blocks, 
    dilation_growth=dilation_growth, 
    n_channels=n_channels,
    latent_dim=latent_dim,
    use_kl=use_kl,
    use_skip=use_skip,
    use_noise=use_noise,
    use_wn=use_wn,
    use_residual=use_residual,
    dilate_conv=dilate_conv,
    use_latent=use_latent,
    activation=activation,
    stride=stride,
    padding=padding
)

# Load the model state
encoder_path = Path('model/checkpoints/encoder.pth')
decoder_path = Path('model/checkpoints/decoder.pth')
encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu'), weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu'), weights_only=True))

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# Initialize PQMF
pqmf = PQMF(100, n_bands)

def process_audio(audio, encoder, decoder, pqmf):
    # Convert to tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    # Pad audio to next power of 2 if n_bands > 1
    if n_bands > 1:
        audio_tensor = center_pad_next_pow_2(audio_tensor)

    # Apply PQMF if n_bands > 1
    if n_bands > 1:
        audio_decomposed = pqmf(audio_tensor)
    else:
        audio_decomposed = audio_tensor

    # Process through the model
    with torch.no_grad():
        if use_kl:
            mu, logvar, encoder_outputs = encoder(audio_decomposed)
            z = encoder.reparameterize(mu, logvar)
        else:
            encoder_outputs = encoder(audio_decomposed)
            z = encoder_outputs.pop()
        encoder_outputs = encoder_outputs[::-1]
        output_decomposed = decoder(z, encoder_outputs)

    # Inverse PQMF if n_bands > 1
    if n_bands > 1:
        output = pqmf.inverse(output_decomposed)
    else:
        output = output_decomposed

    return output.squeeze(0).numpy()

def main():
    # Set the fixed number of samples
    n_samples = 2**19

    # Process an audio file
    input_file = 'mixkit-creature-sad-crying-465.wav'  # Replace with your input file path
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Load audio file
    audio, sample_rate = librosa.load(input_file, sr=None, mono=True)

    # Cut or pad audio to n_samples
    if len(audio) > n_samples:
        audio = audio[:n_samples]
    elif len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)), 'constant')

    # Process audio
    output = process_audio(audio, encoder, decoder, pqmf)

    # Save input audio
    sf.write(output_dir / 'input_fixed_length.wav', audio, sample_rate)

    # Save reconstructed audio
    sf.write(output_dir / 'output_fixed_length.wav', output, sample_rate)

    print(f"Processed audio with {n_samples} samples. Files saved in {output_dir}")

if __name__ == "__main__":
    main()