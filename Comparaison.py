import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from source.network.encoder import EncoderTCN
from source.network.decoder import DecoderTCN
from source.utils import config
from pedalboard.io import AudioFile

class PaperMetricsEvaluator:
    def __init__(self):
        # Initialize the pre-emphasis filter coefficients
        self.pre_emph_coeff = 0.95
        
        # Load the parameters
        self.params = config.Params('params.yaml')
        
        # Initialize encoder and decoder with your parameters
        self.encoder = self._init_encoder()
        self.decoder = self._init_decoder()
        
        # Load model weights
        self._load_models()
        
        # Calculate receptive field
        self.receptive_field = 2**17  # As specified
        
    def _init_encoder(self):
        return EncoderTCN(
            n_inputs=self.params["train"]["n_bands"],
            kernel_size=self.params["train"]["kernel_size"], 
            n_blocks=self.params["train"]["n_blocks"], 
            dilation_growth=self.params["train"]["dilation_growth"], 
            n_channels=self.params["train"]["n_channels"],
            latent_dim=self.params["train"]["latent_dim"],
            use_kl=self.params["train"]["use_kl"],
            use_wn=self.params["train"]["use_wn"],
            use_batch_norm=self.params["train"]["use_batch_norm"],
            use_latent=self.params["train"]["use_latent"],
            dilate_conv=self.params["train"]["dilate_conv"],
            activation=self.params["train"]["activation"],
            stride=self.params["train"]["stride"],
            padding=self.params["train"]["padding"]
        )

    def _init_decoder(self):
        return DecoderTCN(
            n_outputs=self.params["train"]["n_bands"],
            kernel_size=self.params["train"]["kernel_size"],
            n_blocks=self.params["train"]["n_blocks"], 
            dilation_growth=self.params["train"]["dilation_growth"], 
            n_channels=self.params["train"]["n_channels"],
            latent_dim=self.params["train"]["latent_dim"],
            use_kl=self.params["train"]["use_kl"],
            use_skip=self.params["train"]["use_skip"],
            use_noise=self.params["train"]["use_noise"],
            use_wn=self.params["train"]["use_wn"],
            use_residual=self.params["train"]["use_residual"],
            dilate_conv=self.params["train"]["dilate_conv"],
            use_latent=self.params["train"]["use_latent"],
            activation=self.params["train"]["activation"],
            stride=self.params["train"]["stride"],
            padding=self.params["train"]["padding"],
            use_upsampling=self.params["train"]["use_upsampling"]
        )

    def _load_models(self):
        encoder_path = Path('model/checkpoints/encoder.pth')
        decoder_path = Path('model/checkpoints/decoder.pth')
        
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError("Model checkpoint files not found!")
            
        self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
        
        self.encoder.eval()
        self.decoder.eval()

    def pre_emphasis_filter(self, audio_tensor):
        """Apply pre-emphasis filter H(z) = 1 - 0.95z^(-1)"""
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor)
            
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        filtered = torch.zeros_like(audio_tensor)
        filtered[:, 0] = audio_tensor[:, 0]
        for t in range(1, audio_tensor.shape[1]):
            filtered[:, t] = audio_tensor[:, t] - self.pre_emph_coeff * audio_tensor[:, t-1]
        
        return filtered

    def calculate_paper_metrics(self, output, target):
        """Calculate metrics as defined in the paper"""
        # Trim target to match output length considering receptive field
        target = target[..., self.receptive_field-1:]
        
        # Ensure output and target have the same length
        min_length = min(output.shape[-1], target.shape[-1])
        output = output[..., :min_length]
        target = target[..., :min_length]
        
        # Apply pre-emphasis filter for MAE calculation
        output_emph = self.pre_emphasis_filter(output)
        target_emph = self.pre_emphasis_filter(target)
        
        # Calculate MAE in time domain
        mae = torch.mean(torch.abs(output_emph - target_emph)).item()
        
        # Calculate MSE in frequency domain using 4096-point FFT
        output_fft = torch.fft.fft(output, n=4096)
        target_fft = torch.fft.fft(target, n=4096)
        
        output_mag = torch.log(torch.abs(output_fft) + 1e-8)
        target_mag = torch.log(torch.abs(target_fft) + 1e-8)
        
        mse = torch.mean((output_mag - target_mag) ** 2).item()
        
        # Calculate combined loss
        combined_loss = mae + 1e-4 * mse
        
        return {
            'mae': mae,
            'mse': mse,
            'combined_loss': combined_loss
        }

    def evaluate_model(self, input_path, target_path):
        """Evaluate model using a pair of input and target audio files"""
        # Load input audio
        with AudioFile(input_path) as f:
            input_audio = torch.tensor(f.read(f.frames)).float()
            sample_rate = f.samplerate
            if input_audio.dim() > 1:
                input_audio = input_audio.mean(dim=0)
            input_audio = input_audio.unsqueeze(0).unsqueeze(0)
        
        # Load target audio
        with AudioFile(target_path) as f:
            target_audio = torch.tensor(f.read(f.frames)).float()
            if target_audio.dim() > 1:
                target_audio = target_audio.mean(dim=0)
            target_audio = target_audio.unsqueeze(0).unsqueeze(0)
        
        # Process through the encoder-decoder
        with torch.no_grad():
            encoder_outputs = self.encoder(input_audio)
            z = encoder_outputs.pop()
            output = self.decoder(z, encoder_outputs[::-1])
        
        # Calculate metrics
        metrics = self.calculate_paper_metrics(
            output.squeeze().cpu(), 
            target_audio.squeeze().cpu()
        )
        
        return metrics, output, sample_rate

def main():
    # Initialize evaluator
    evaluator = PaperMetricsEvaluator()
    
    # Paths to your audio files
    input_file = 'input_fixed_length.wav'
    target_file = 'result.wav'  # Your wet/processed target audio
    
    # Evaluate
    try:
        metrics, output, sr = evaluator.evaluate_model(input_file, target_file)
        
        print("Model Evaluation Results:")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"Combined Loss: {metrics['combined_loss']:.6f}")
        
        print("\nPaper Benchmarks:")
        print("Plate Reverb - MAE: 0.00214, MSE: 7.75815, Loss: 0.00292")
        print("Spring Reverb - MAE: 0.00366, MSE: 9.43629, Loss: 0.00461")
        
        # Optionally save the output
        output_path = 'output_evaluated.wav'
        sf.write(output_path, output.squeeze().cpu().numpy(), sr)
        print(f"\nOutput audio saved to {output_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()