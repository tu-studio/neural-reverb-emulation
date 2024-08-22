import torch
from network.ravepqmf import PQMF, center_pad_next_pow_2
from utils import config
import os

def evaluate(encoder, decoder, test_loader, criterion, tensorboard_writer, device='cpu', n_bands=64, use_kl=False, sample_rate=44100):
    encoder.to(device)
    decoder.to(device)
    
    # Set model to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Initialize PQMF
    pqmf = PQMF(100, n_bands).to(device)
    
    test_loss = 0
    test_kl_div = 0
    test_criterion = 0
    
    with torch.no_grad():
        for batch, (dry_audio, wet_audio) in enumerate(test_loader):
            # Reshape audio
            dry_audio = dry_audio[0:1, :]
            wet_audio = wet_audio[0:1, :]  
           
            dry_audio = dry_audio.view(1, 1, -1)
            wet_audio = wet_audio.view(1, 1, -1)
            wet_audio = wet_audio[:,:, :dry_audio.shape[-1]]
            
            dry_audio, wet_audio = dry_audio.to(device), wet_audio.to(device)

            # Pad both dry and wet audio to next power of 2
            dry_audio = center_pad_next_pow_2(dry_audio)
            wet_audio = center_pad_next_pow_2(wet_audio)
            
            # Apply PQMF to input
            dry_audio_decomposed = pqmf(dry_audio)
            wet_audio_decomposed = pqmf(wet_audio)

            audio_difference_decomposed = wet_audio_decomposed - dry_audio_decomposed
            audio_difference = wet_audio - dry_audio

            # Forward pass through encoder
            encoder_outputs = []
            x = dry_audio_decomposed
            for block in encoder.blocks:
                x = block(x)
                encoder_outputs.append(x)
    
            # Get the final encoder output
            z = encoder_outputs.pop()

            # Reverse the list of encoder outputs for the decoder
            encoder_outputs = encoder_outputs[::-1]
            encoder_outputs.append(dry_audio_decomposed)

            # Forward pass through encoder
            if use_kl:
                mu, logvar = encoder(dry_audio_decomposed)
                z = encoder.reparameterize(mu, logvar)
                kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[-1]
                test_kl_div += kl_div

            # Forward pass through decoder
            net_outputs_decomposed = decoder(z, encoder_outputs)

            net_outputs = pqmf.inverse(net_outputs_decomposed)

            # Trim outputs to original length
            original_length = dry_audio.shape[-1]
            net_outputs = net_outputs[..., :original_length]
            wet_audio = wet_audio[..., :original_length]
            dry_audio = dry_audio[..., :original_length]

            # Compute loss
            loss = criterion(net_outputs + dry_audio, wet_audio)
            if use_kl:
                loss += kl_div
          
            # Output
            output = net_outputs + dry_audio

            # Accumulate losses
            test_loss += loss 
            test_criterion += loss

            # Log audio samples (for the first batch only)
            if batch == 0:
                tensorboard_writer.add_audio("Audio/TCN_Input", dry_audio.cpu().squeeze(0), 0, sample_rate=sample_rate)
                tensorboard_writer.add_audio("Audio/TCN_Target", wet_audio.cpu().squeeze(0), 0, sample_rate=sample_rate)
                tensorboard_writer.add_audio("Audio/TCN_output", output.cpu().squeeze(0), 0, sample_rate=sample_rate)

    # Calculate average losses
    test_avg_loss = test_loss / len(test_loader)
    test_avg_criterion = test_criterion / len(test_loader)
    if use_kl:
        test_avg_kl_div = test_kl_div / len(test_loader)

    # Log final metrics
    tensorboard_writer.add_scalar("Test Loss/Overall", test_avg_loss, 0)
    tensorboard_writer.add_scalar("Test Loss/Criterion", test_avg_criterion, 0)
    if use_kl:
        tensorboard_writer.add_scalar("Test Loss/KL Divergence", test_avg_kl_div, 0)
    
    tensorboard_writer.flush()
    tensorboard_writer.step()
    tensorboard_writer.close()

    print(f'Test Loss: {test_avg_loss}')
    if use_kl:
        print(f'Test KL Divergence: {test_avg_kl_div}')
    print(f'Test Criterion: {test_avg_criterion}')


