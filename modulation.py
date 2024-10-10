import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from source.network.encoder import EncoderTCN
from source.network.decoder import DecoderTCN
from source.utils import config
from pedalboard.io import AudioFile
from sklearn.decomposition import PCA
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import threading
import shutil
import os

app = Flask(__name__)

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

# Process through the encoder
with torch.no_grad():
    encoder_outputs = encoder(audio_tensor)
    z = encoder_outputs.pop()

# Perform PCA
z_pca = z.squeeze().cpu().numpy().T
pca = PCA(n_components=0.95)  # Keep 95% of variance
z_pca_reduced = pca.fit_transform(z_pca)

# Global variables
is_processing = False
fader_values = [0] * z_pca_reduced.shape[1]  # Initialize faders for all PCA components

def process_audio():
    global is_processing
    if is_processing:
        return "Processing in progress"
    
    is_processing = True

    # Generate output before changes
    z_reconstructed = pca.inverse_transform(z_pca_reduced)
    z_reconstructed = torch.from_numpy(z_reconstructed.T).float().unsqueeze(0).to(z.device)
    with torch.no_grad():
        output_before = decoder(z_reconstructed, encoder_outputs[::-1])

    # Modulate all PCA channels based on fader values
    z_pca_modulated = z_pca_reduced.copy()
    for i in range(z_pca_reduced.shape[1]):
        z_pca_modulated[:, i] += fader_values[i]

    z_reconstructed_modulated = pca.inverse_transform(z_pca_modulated)
    z_reconstructed_modulated = torch.from_numpy(z_reconstructed_modulated.T).float().unsqueeze(0).to(z.device)

    with torch.no_grad():
        output_modulated = decoder(z_reconstructed_modulated, encoder_outputs[::-1])

    # Convert outputs to numpy array and write to files
    output_before = output_before.squeeze().cpu().numpy()
    output_modulated = output_modulated.squeeze().cpu().numpy()
    sf.write(output_dir / 'output_before.wav', output_before, sample_rate)
    sf.write(output_dir / 'output_modulated.wav', output_modulated, sample_rate)

    # Copy the input file to the output directory for easy access
    shutil.copy(str(input_file), str(output_dir / 'input.wav'))

    is_processing = False
    return "Processing complete. You can now play the input, output before changes, and output after changes."

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Audio Modulation Interface</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .slider-container { margin-bottom: 20px; display: flex; align-items: center; }
                .slider-container label { width: 100px; }
                input[type="range"] { width: 70%; margin: 0 10px; }
                .value-display { width: 50px; text-align: right; }
                .audio-container { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Audio Modulation Interface</h1>
            <div id="sliders"></div>
            <button id="processButton">Process Audio</button>
            <p id="status"></p>
            <div class="audio-container">
                <h3>Input Audio</h3>
                <audio id="inputAudio" controls>
                    <source src="/audio/input.wav" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <div class="audio-container">
                <h3>Output Before Changes</h3>
                <audio id="outputBeforeAudio" controls>
                    <source src="/audio/output_before.wav" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <div class="audio-container">
                <h3>Output After Changes</h3>
                <audio id="outputAfterAudio" controls>
                    <source src="/audio/output_modulated.wav" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            <script>
                $(document).ready(function() {
                    // Dynamically create sliders
                    var sliderCount = {{ z_pca_reduced.shape[1] }};
                    var sliderContainer = $('#sliders');
                    for (var i = 0; i < sliderCount; i++) {
                        sliderContainer.append(`
                            <div class="slider-container">
                                <label for="slider${i}">Channel ${i + 1}</label>
                                <input type="range" id="slider${i}" min="-10" max="10" step="0.1" value="0">
                                <span class="value-display" id="value${i}">0.0</span>
                            </div>
                        `);
                    }

                    $('input[type="range"]').on('input', function() {
                        var id = $(this).attr('id');
                        var value = parseFloat($(this).val()).toFixed(1);
                        $('#value' + id.replace('slider', '')).text(value);
                    });

                    $('#processButton').click(function() {
                        var values = [];
                        $('input[type="range"]').each(function() {
                            values.push(parseFloat($(this).val()));
                        });
                        $.ajax({
                            url: '/process',
                            method: 'POST',
                            contentType: 'application/json',
                            data: JSON.stringify({values: values}),
                            success: function(response) {
                                $('#status').text(response.status);
                                // Reload the audio elements
                                $('#inputAudio').get(0).load();
                                $('#outputBeforeAudio').get(0).load();
                                $('#outputAfterAudio').get(0).load();
                            }
                        });
                    });
                });
            </script>
        </body>
        </html>
    ''', z_pca_reduced=z_pca_reduced)

@app.route('/process', methods=['POST'])
def process():
    global fader_values
    fader_values = request.json['values']
    result = process_audio()
    return jsonify({"status": result})

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(str(output_dir), filename)

if __name__ == '__main__':
    app.run(debug=True)