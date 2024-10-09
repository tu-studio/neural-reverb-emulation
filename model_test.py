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
from source.network.metrics import spectral_distance, single_stft_loss, fft_loss
from pedalboard.io import AudioFile

# Load the parameters
params = config.Params('params.yaml')

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        return audio / max_val
    return audio

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

# Initialize PQMF
pqmf = PQMF(100, n_bands)

# Define loss function (you may need to adjust this based on your specific criterion)
criterion = torch.nn.MSELoss()

def process_audio_chunk(audio_chunk, model, pqmf):
    # Convert to tensor and add batch dimension
    audio_chunk = torch.from_numpy(audio_chunk).float().unsqueeze(0)
    
    # Pad audio to next power of 2
    audio_chunk = center_pad_next_pow_2(audio_chunk)

    # Apply PQMF
    audio_chunk_decomposed = pqmf(audio_chunk)

    # Process through the model
    with torch.no_grad():
        if use_kl:
            mu, logvar, encoder_outputs = encoder(audio_chunk_decomposed)
            z = encoder.reparameterize(mu, logvar)
        else:
            encoder_outputs = encoder(audio_chunk_decomposed)
            z = encoder_outputs.pop()
        encoder_outputs = encoder_outputs[::-1]
        output_decomposed = decoder(z, encoder_outputs)

    # Inverse PQMF
    output = pqmf.inverse(output_decomposed)

    return output.squeeze(0).numpy(), output_decomposed

def calculate_loss(output_decomposed, target_decomposed):
    return criterion(output_decomposed, target_decomposed).item()

def process_full_audio(audio, pqmf):
    # Convert to tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    
    # Pad audio to next power of 2
    # audio_tensor = center_pad_next_pow_2(audio_tensor)

    # Apply PQMF
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

    # Inverse PQMF
    output = output_decomposed

    return output.squeeze(0).numpy(), output_decomposed

def main():
    # Process an audio file
    input_file = 'output/input_fixed_length.wav'  # Replace with your input file path
    output_dir = Path('output_chunks')
    output_dir.mkdir(exist_ok=True)

    # Load audio file
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

        # If stereo, convert to mono by taking the mean of both channels
        if audio.ndim > 1 and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
        else:
            audio = audio.squeeze() 

    # manage clipping
    audio = normalize_audio(audio)

    print(audio.shape)

    # Process full audio
    full_output, full_output_decomposed = process_full_audio(audio, pqmf)
    
    # Calculate loss for full audio
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    # audio_tensor = center_pad_next_pow_2(audio_tensor)
    audio_decomposed = pqmf(audio_tensor)
    full_loss = calculate_loss(full_output_decomposed, audio_decomposed)
    
    print(f"Loss for full audio: {full_loss:.6f}")

    # Save full input audio
    sf.write(output_dir / 'full_input.wav', audio, sample_rate)

    # Save full reconstructed audio
    sf.write(output_dir / 'full_output.wav', full_output, sample_rate)

    # # Define chunk size
    # chunk_size = 524288

    # # Process audio in chunks
    # total_loss = 0
    # num_chunks = 0

    # for i, start in enumerate(range(0, len(audio), chunk_size)):
    #     end = start + chunk_size
    #     chunk = audio[start:end]

    #     # If the last chunk is smaller than chunk_size, pad it
    #     if len(chunk) < chunk_size:
    #         chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

    #     # Process the chunk
    #     output, output_decomposed = process_audio_chunk(chunk, model, pqmf)

    #     # Calculate loss
    #     chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
    #     chunk_tensor = center_pad_next_pow_2(chunk_tensor)
    #     chunk_decomposed = pqmf(chunk_tensor)
    #     loss = calculate_loss(output_decomposed, chunk_decomposed)
        
    #     total_loss += loss
    #     num_chunks += 1
    #     print(f"Loss for chunk {i}: {loss:.6f}")

    #     # Save input chunk
    #     input_chunk_file = output_dir / f'input_chunk_{i}.wav'
    #     sf.write(input_chunk_file, chunk, sample_rate)

    #     # Save reconstructed chunk
    #     output_chunk_file = output_dir / f'output_chunk_{i}.wav'
    #     sf.write(output_chunk_file, output, sample_rate)

    # # Calculate and print average loss
    # avg_loss = total_loss / num_chunks
    # print(f"Average loss for chunks: {avg_loss:.6f}")

    # print(f"Processed full audio and {num_chunks} chunks. All files saved in {output_dir}")

if __name__ == "__main__":
    main()