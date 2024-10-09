import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from source.network.encoder import EncoderTCN
from source.network.decoder import DecoderTCN
from source.utils import config
from pedalboard.io import AudioFile
from sklearn.decomposition import PCA

# Load the parameters
params = config.Params('params.yaml')

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 1.0 else audio

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
encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

encoder.eval()
decoder.eval()

input_file = 'output/input_fixed_length.wav'  
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Load audio file
with AudioFile(input_file) as f:
    audio = f.read(f.frames)
    sample_rate = f.samplerate

# Convert to mono if stereo
audio = np.mean(audio, axis=0) if audio.ndim > 1 and audio.shape[0] > 1 else audio.squeeze()

# Normalize audio
audio = normalize_audio(audio)

# Convert to tensor
audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

# Process through the model
with torch.no_grad():
    encoder_outputs = encoder(audio_tensor)
    z = encoder_outputs.pop()

    # Perform PCA
    z_pca = z.squeeze().cpu().numpy().T
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    z_pca_reduced = pca.fit_transform(z_pca)
    z_reconstructed = pca.inverse_transform(z_pca_reduced)
    z_reconstructed = torch.from_numpy(z_reconstructed.T).float().unsqueeze(0).to(z.device)

    # Decode
    output = decoder(z_reconstructed, encoder_outputs[::-1])

# Convert output to numpy array and write to file
output = output.squeeze().cpu().numpy()
sf.write(output_dir / 'full_output_PCA.wav', output, sample_rate)

print("Processing complete. Check the output directory for the processed audio file.")

# Optional: Print PCA information
n_components = pca.n_components_
print(f"Number of PCA components retained: {n_components}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")