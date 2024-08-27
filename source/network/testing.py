import torch
from network.ravepqmf import PQMF, center_pad_next_pow_2
from tqdm import tqdm

def test(encoder, decoder, test_loader, criterion, tensorboard_writer, device='cpu', n_bands=64, use_kl=False, sample_rate=44100):
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
    
    # Calculate total number of batches
    total_batches = len(test_loader)

    # Create a progress bar for the testing process
    progress_bar = tqdm(total=total_batches, desc="Testing Progress")

    with torch.no_grad():
        for batch, (dry_audio, wet_audio) in enumerate(test_loader):
            dry_audio = dry_audio.to(device)
            wet_audio = wet_audio.to(device)

            # Pad both dry and wet audio to next power of 2
            dry_audio = center_pad_next_pow_2(dry_audio)
            wet_audio = center_pad_next_pow_2(wet_audio)
            
            # Apply PQMF to input
            dry_audio_decomposed = pqmf(dry_audio)
            wet_audio_decomposed = pqmf(wet_audio)

            # Forward pass through encoder
            if use_kl:
                mu, logvar, encoder_outputs = encoder(dry_audio_decomposed)
                z = encoder.reparameterize(mu, logvar)
                kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[-1]
                test_kl_div += kl_div
            else:
                encoder_outputs = encoder(dry_audio_decomposed)
                z = encoder_outputs.pop()

            # Reverse the list of encoder outputs for the decoder
            encoder_outputs = encoder_outputs[::-1]

            # Forward pass through decoder
            net_outputs_decomposed = decoder(z, encoder_outputs)

            output = pqmf.inverse(net_outputs_decomposed)
            dry = pqmf.inverse(dry_audio_decomposed)
            wet = pqmf.inverse(wet_audio_decomposed)

            # Compute loss
            loss = criterion(output + dry, wet)
            if use_kl:
                loss += kl_div
          
            # Accumulate losses
            test_loss += loss 
            test_criterion += loss

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

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
    progress_bar.close()

    print(f'Test Loss: {test_avg_loss:.4f}')
    if use_kl:
        print(f'Test KL Divergence: {test_avg_kl_div:.4f}')
    print(f'Test Criterion: {test_avg_criterion:.4f}')