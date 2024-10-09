import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from source.network.encoder import EncoderTCN
from source.network.decoder import DecoderTCN
from source.network.CombinedModels import CombinedEncoderDecoder
from source.utils import config
from source.network.ravepqmf import PQMF, center_pad_next_pow_2
from source.network.metrics import spectral_distance, single_stft_loss, fft_loss
from pedalboard.io import AudioFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


input_file = 'output/input_fixed_length.wav'  
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

audio_tensor = torch.from_numpy(audio)
audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

# Apply PQMF
audio_decomposed = audio_tensor

# Process through the model
with torch.no_grad():
    encoder_outputs = encoder(audio_decomposed)
    z = encoder_outputs.pop()

print(z.shape)
z_pca = z.squeeze()
print(z.shape)
z_pca = z_pca.numpy()

# Calculate variance along each dimension
variances = np.var(z_pca, axis=1)

print(variances)

# Sort variances in descending order
sorted_indices = np.argsort(variances)[::-1]
sorted_variances = variances[sorted_indices]


total_variance = np.sum(variances)
cumulative_variance_ratio = np.cumsum(sorted_variances) / total_variance

n_dimensions_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of dimensions needed to explain 95% of variance: {n_dimensions_95}")



n_top_dimensions = 10 
print("Top 10 most important dimensions:")
for i, idx in enumerate(sorted_indices[:n_top_dimensions], 1):
    print(f"{i}. Dimension index: {idx}, Variance: {variances[idx]:.4f}")


z_pca = z_pca.T
#Perform PCA
pca = PCA()
pca_result = pca.fit_transform(z_pca)

print(pca_result.shape)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

print("explained variance ratio:", pca.explained_variance_ratio_)

# Determine number of dimensions for 95% variance explained
n_dimensions_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of latent dimensions needed to explain 95% of variance: {n_dimensions_95}")

print("pca components:", pca.components_)

# Analyze the top 10 most important latent dimensions
n_top_dimensions = 10
dimension_importance = np.sum(np.abs(pca.components_), axis=0)
top_dimension_indices = np.argsort(dimension_importance)[-n_top_dimensions:][::-1]

print(top_dimension_indices)

print("\nTop 10 most important latent dimensions:")
for i, idx in enumerate(top_dimension_indices, 1):
    print(f"{i}. Dimension index: {idx}, Importance: {dimension_importance[idx]:.4f}")


# Perform PCA with the determined number of dimensions
pca = PCA(n_components=n_dimensions_95)
pca_result = pca.fit_transform(z_pca)

z_reconstructed = pca.inverse_transform(pca_result)

encoder_outputs = encoder_outputs[::-1]

print(z_reconstructed.shape)
z_reconstructed = z_reconstructed.T
z_reconstructed = torch.from_numpy(z_reconstructed).unsqueeze(0)

print(z_reconstructed.shape)

output = decoder(z_reconstructed, encoder_outputs)

output = output.squeeze(0).squeeze(0).numpy()

sf.write(output_dir / 'full_output_3.wav', output, sample_rate)
print(pca_result.shape)
print(z_reconstructed.shape)