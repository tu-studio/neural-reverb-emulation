import torch
import librosa
import soundfile as sf
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the ONNX model
model_path = Path('model/exports/model.onnx')
ort_session = onnxruntime.InferenceSession(str(model_path))

# Get model input name and shape
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
print(f"Model input name: {input_name}")
print(f"Model input shape: {input_shape}")

def load_audio(file_path, target_length=None):
    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    
    # Adjust length if needed
    if target_length is not None:
        if len(waveform) < target_length:
            waveform = librosa.util.fix_length(waveform, target_length)
        else:
            waveform = waveform[:target_length]
    
    return waveform, sample_rate

def process_audio(waveform, ort_session, input_name):
    # Prepare input (ensure it's 3D for ONNX input: [batch, channel, time])
    model_input = waveform[np.newaxis, np.newaxis, :]
    
    # Run inference
    ort_inputs = {input_name: model_input.astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Convert output back to 1D numpy array
    processed_audio = ort_outputs[0].squeeze()
    
    return processed_audio

def plot_waveforms(original, processed):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(original)
    ax1.set_title('Original Waveform')
    ax1.set_ylim([-1, 1])
    
    ax2.plot(processed)
    ax2.set_title('Processed Waveform')
    ax2.set_ylim([-1, 1])
    
    plt.tight_layout()
    plt.show()

def save_audio(waveform, sample_rate, file_path):
    sf.write(file_path, waveform, sample_rate)
    print(f"Saved processed audio to {file_path}")

def main():
    # Load your audio file
    audio_path = 'hypno project- save 3.wav'  # Replace with your audio file path
    waveform, sample_rate = load_audio(audio_path)

    print(f"Loaded audio shape: {waveform.shape}, Sample rate: {sample_rate}")
    
    # Process audio with the model
    processed_waveform = process_audio(waveform, ort_session, input_name)
    print(f"Processed audio shape: {processed_waveform.shape}")

    # Plot waveforms
    plot_waveforms(waveform, processed_waveform)

    # Save the processed audio
    output_path = 'processed_audio.wav'  # Replace with your desired output path
    save_audio(processed_waveform, sample_rate, output_path)

    print("Audio processing complete. Check the output file and the generated plot.")

if __name__ == "__main__":
    main()