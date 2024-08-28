import os
import random
import numpy as np
from pedalboard import Pedalboard, load_plugin
from pedalboard.io import AudioFile
from network.dataset import AudioDataset
from utils import config

# Load the hyperparameters from the params yaml file into a Dictionary
params = config.Params('params.yaml')

def load_reverb():
    plate_reverb = load_plugin(params["preprocess"]["plate_reverb_path"])
    plate_reverb.blend_dry_wet = 1
    return Pedalboard([plate_reverb])

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        return audio / max_val
    return audio

def apply_zero_padding(audio, pad_length):
    # Check if padding is necessary
    if pad_length <= 0:
        return audio

    # Create a padding tuple that pads only the last dimension
    pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, pad_length)]

    return np.pad(audio, pad_width, mode='constant', constant_values=0)

def apply_zero_padding_random(audio, pad_length, safe = False):
    # Check if padding is necessary
    if pad_length <= 0:
        return audio

    # Determine the padding strategy
    padding_strategy = np.random.choice(['end', 'beginning', 'random'], p=[0.2, 0.2, 0.6])
    
    # Create a padding tuple that pads only the last dimension
    pad_width = [(0, 0)] * (audio.ndim - 1)
    
    if padding_strategy == 'end':
        pad_width.append((0, pad_length))
    elif padding_strategy == 'beginning':
        if safe:
            pad_width.append((pad_length, pad_length))
        else:
            pad_width.append((pad_length, 0))
    else:  # 'random'
        start_pad = np.random.randint(0, pad_length + 1)
        end_pad = pad_length - start_pad
        if safe:
            end_pad = pad_length
        pad_width.append((start_pad, end_pad))

    return np.pad(audio, pad_width, mode='constant', constant_values=0)

def agglomerate_segments(segments, target_length):
    combined = np.hstack(segments)
    if len(combined) > target_length:
        return combined[:target_length]
    elif len(combined) < target_length:
        return apply_zero_padding(combined, target_length - len(combined))
    else:
        return combined

def process_audio_with_reverb(audio, board, sample_rate, chunk_size):
    effected_audio = []
    for i in range(0, audio.shape[-1], chunk_size):
        chunk = audio[..., i:i + chunk_size]
        effected = board(chunk, sample_rate=sample_rate, reset=False)
        effected_audio.append(effected)
    return np.hstack(effected_audio)

def calculate_max_tail_length(board):
    silences = []
    for _ in range(params["preprocess"]["num_noises"]):
        noise = np.random.normal(-1, 1, int(params["preprocess"]["noise_duration"] * params["general"]["sample_rate"]))
        silence = np.zeros((int(params["preprocess"]["noise_duration"] * params["general"]["sample_rate"])))
        audio = np.concatenate((noise, silence))
        effected_audio = process_audio_with_reverb(audio, board, params["general"]["sample_rate"], params["preprocess"]["board_chunk_size"])
        
        i = 0
        while i < np.shape(effected_audio)[-1]:
            chunk = effected_audio[..., i:i + params["preprocess"]["sliding_mean_length"]]
            if np.mean(abs(chunk)) < 1e-5 and i > params["general"]["sample_rate"] * params["preprocess"]["noise_duration"]:
                break
            i += 1
        silences.append(i - params["general"]["sample_rate"] * params["preprocess"]["noise_duration"])
    max_tail = np.max(silences)

    return max_tail

def apply_fade(audio, fade_length, fade_type='in'):
    if fade_length > audio.shape[-1]:
        if fade_type == 'out':
            raise ValueError('Fade out length cannot be greater than audio length')
        fade_length = audio.shape[-1]
    
    fade = np.linspace(0, 1, fade_length) if fade_type == 'in' else np.linspace(1, 0, fade_length)
    
    # Reshape fade to match audio channels
    fade = fade.reshape(1, -1) if audio.ndim == 2 else fade
    
    faded_audio = audio.copy()
    if fade_type == 'in':
        faded_audio[..., :fade_length] *= fade
    else:
        faded_audio[..., -fade_length:] *= fade
    return faded_audio

def process_files(directory, max_tail_length, model_chunk_size, destination_directory, short_directory):
    os.makedirs(destination_directory, exist_ok=True)
    os.makedirs(short_directory, exist_ok=True)

    segment_length = model_chunk_size - max_tail_length

    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(('.wav', '.aif')):
                file_path = os.path.join(root, file_name)

                dir_name, file_name = os.path.split(file_path)
                base_name, ext = os.path.splitext(file_name)

                with AudioFile(file_path) as f:
                    audio = f.read(int(f.samplerate * f.duration))
                    num_segments = audio.shape[-1] // segment_length
                    num_channels = f.num_channels
                    sample_rate = f.samplerate

                if num_segments == 0:
                    end = 0
                    i = 0

                for i in range(num_segments):
                    start = i * segment_length
                    end = start + segment_length
                    segment = audio[:,start:end]

                    # Apply fade in and fade out
                    if random.random() < 0.9:
                        fade_in_length = int(random.uniform(0.05, 0.1) * sample_rate)
                        fade_out_length = int(random.uniform(0.05, 0.1) * sample_rate)
                        if i > 0:
                            segment = apply_fade(segment, fade_in_length, 'in')
                        segment = apply_fade(segment, fade_out_length, 'out')

                    # Apply zero padding
                    padded_segment = apply_zero_padding_random(segment, max_tail_length)

                    # Save the segment with the destination directory
                    segment_name = f"segment_{i}.wav"
                    destination_file_name = base_name +segment_name
                    segment_path = os.path.join(destination_directory, destination_file_name)
                    with AudioFile(segment_path, 'w', sample_rate, num_channels) as f:
                        f.write(padded_segment)

                # Process the last segment
                last_segment = audio[:,end:]
                if random.random() < 0.9:
                    fade_in_length = int(random.uniform(0.05, 0.1) * sample_rate)
                    last_segment = apply_fade(last_segment, fade_in_length, 'in')
                padded_last_segment = apply_zero_padding_random(last_segment, max_tail_length, True)
                
                # Save the segment with the destination directory
                segment_name = f"segment_{i}.wav"
                destination_file_name = base_name +segment_name
                segment_path = os.path.join(short_directory, destination_file_name)
                with AudioFile(segment_path, 'w', sample_rate, num_channels) as f:
                    f.write(padded_last_segment)

def agglomerate_short_segments(short_directory, dry_directory, model_chunk_size):
    short_segments = []
    agglomerated_count = 0

    for root, _, files in os.walk(short_directory):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)

                with AudioFile(file_path) as f:
                    audio = f.read(f.frames)
                    short_segments.append(audio)
                    num_channels = f.num_channels

                if sum(len(seg) for seg in short_segments) >= model_chunk_size:
                    agglomerated = agglomerate_segments(short_segments, model_chunk_size)
                    agglomerated_file_name = f"agglomerated_{agglomerated_count}.wav"
                    agglomerated_path = os.path.join(dry_directory, agglomerated_file_name)
                    with AudioFile(agglomerated_path, 'w', params["general"]["sample_rate"], num_channels) as f:
                        f.write(agglomerated)
                    agglomerated_count += 1
                    short_segments = []

def apply_wet_processing(dry_directory, wet_directory, board):
    os.makedirs(wet_directory, exist_ok=True)

    for root, _, files in os.walk(dry_directory):
        for file_name in files:
            if file_name.endswith('.wav'):
                file_path = os.path.join(root, file_name)

                with AudioFile(file_path) as f:
                    audio = f.read(f.frames)
                    sample_rate = f.samplerate

                # manage clipping
                audio = normalize_audio(audio)

                wet_audio = process_audio_with_reverb(audio, board, sample_rate, params["preprocess"]["board_chunk_size"])
                wet_audio = normalize_audio(wet_audio)
                
                wet_file_path = os.path.join(wet_directory, file_name)

                with AudioFile(wet_file_path, 'w', sample_rate, wet_audio.shape[0]) as f:
                    f.write(wet_audio)

def explore_directory(directory, extension=".wav"):
    files_list = []
    for root, directories, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(extension):
                files_list.append(os.path.join(root, file_name))
    return files_list


def save_data_to_pt():
    if not os.path.exists('data/processed'):
        os.mkdir('data/processed')   
    dry_directory = params["preprocess"]["dry_output_directory"]
    wet_directory = params["preprocess"]["wet_output_directory"]

    dry_audio_files = explore_directory(dry_directory)
    wet_audio_files = explore_directory(wet_directory)

    # Create the dataset and save to pt
    AudioDataset.save_to_pt(dry_audio_files, wet_audio_files, 'data/processed/preprocessed_data.pt')

def main():
    board = load_reverb()

    # 0. Calculate max tail length
    max_tail_length = calculate_max_tail_length(board)
    print(f"Max tail length: {max_tail_length}")

    # 1. Define Chunk size
    if params["general"]["input_size"] < max_tail_length:
        raise ValueError("Model input size must be greater or equal to the max tail length")

    model_chunk_size = params["general"]["input_size"]
   
    # 2. Truncate data to the model and 3. apply zero padding
    process_files(params["preprocess"]["input_directory"], max_tail_length, model_chunk_size, 
                  params["preprocess"]["dry_output_directory"], params["preprocess"]["short_output_directory"])
    
    # 5. Agglomerate short segments
    agglomerate_short_segments(params["preprocess"]["short_output_directory"], 
                               params["preprocess"]["dry_output_directory"], model_chunk_size)
   
    # 4. Apply pedal board to dry audio
    apply_wet_processing(params["preprocess"]["dry_output_directory"], 
                         params["preprocess"]["wet_output_directory"], board)

    # 6. Save to pt
    save_data_to_pt()

    # 7. Delete temporary files
    os.system(f"rm -r {params['preprocess']['short_output_directory']}")

if __name__ == "__main__":
    main()