import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from source.network.encoder import EncoderTCN
from source.network.decoder import DecoderTCN
from source.network.CombinedModels import CombinedEncoderDecoder
from source.utils import config

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
encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

# Combine encoder and decoder
model = CombinedEncoderDecoder(encoder, decoder)
model.eval()  # Set the model to evaluation mode

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        return audio / max_val
    return audio

def process_audio(file_path, model):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)

    # Normalize audio
    audio = normalize_audio(audio)

    # Prepare input for the model
    input_audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Process audio through the model
    with torch.no_grad():
        if use_kl:
            mu, logvar, encoder_outputs = encoder(input_audio)
            z = encoder.reparameterize(mu, logvar)
        else:
            encoder_outputs = encoder(input_audio)
            z = encoder_outputs.pop()
        encoder_outputs = encoder_outputs[::-1]
        output_audio = decoder(z, encoder_outputs)

    return audio, output_audio.numpy().squeeze(), sample_rate

def plot_waveforms(dry, reconstructed, sample_rate):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    time = np.arange(len(dry)) / sample_rate
    ax1.plot(time, dry)
    ax1.set_title('Dry Audio')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')

    ax2.plot(time[:len(reconstructed)], reconstructed)
    ax2.set_title('Reconstructed Audio')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def main():
    # Process an audio file
    input_file = 'mixkit-creature-sad-crying-465.wav'  # Replace with your input file path
    dry_audio, reconstructed_audio, sample_rate = process_audio(input_file, model)

    # Plot waveforms
    plot_waveforms(dry_audio, reconstructed_audio, sample_rate)

    # Save the reconstructed audio
    output_file = 'output_audio.wav'  # Replace with your desired output file path
    sf.write(output_file, reconstructed_audio, sample_rate)

if __name__ == "__main__":
    main()