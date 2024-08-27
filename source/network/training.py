import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from network.ravepqmf import PQMF, center_pad_next_pow_2
from utils import config
import os
from tqdm import tqdm

def train(encoder, decoder, train_loader, val_loader, criterion, optimizer, scheduler, tensorboard_writer, num_epochs=25, device='cpu', n_bands=64, use_kl=False, sample_rate=44100):
    encoder.to(device)
    if decoder:
        decoder.to(device)

    # Initialize PQMF
    pqmf = PQMF(100, n_bands).to(device)

    # Calculate total number of batches
    total_batches = num_epochs * len(train_loader)

    # Create a progress bar for the entire training process
    progress_bar = tqdm(total=total_batches, desc="Training Progress")

    for epoch in range(num_epochs):
        #Train mode
        encoder.train()
        if decoder:
            decoder.train()

        #initialize epoch losses
        train_epoch_loss = 0
        train_epoch_kl_div = 0
        train_epoch_criterion = 0

        for batch, (dry_audio, wet_audio) in enumerate(train_loader):
            dry_audio = dry_audio.to(device)
            wet_audio = wet_audio.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Pad both dry and wet audio to next power of 2
            dry_audio = center_pad_next_pow_2(dry_audio)
            wet_audio = center_pad_next_pow_2(wet_audio)

            # Apply PQMF to input
            dry_audio_decomposed = pqmf(dry_audio)
            wet_audio_decomposed = pqmf(wet_audio)

            # Throw error if wet audio is longer than dry audio
            if wet_audio_decomposed.shape[-1] != dry_audio_decomposed.shape[-1]:
                raise ValueError(f"Wet audio is not the same length than dry audio: {wet_audio_decomposed.shape[-1]} vs {dry_audio_decomposed.shape[-1]}")
        
            dry_audio_decomposed, wet_audio_decomposed = dry_audio_decomposed.to(device), wet_audio_decomposed.to(device)
    
            # Forward pass
            if decoder:
                # Encoder-Decoder architecture
                if use_kl:
                    mu, logvar, encoder_outputs = encoder(dry_audio_decomposed)
                    z = encoder.reparameterize(mu, logvar)
                    kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[-1]
                    train_epoch_kl_div += kl_div
                else:
                    encoder_outputs = encoder(dry_audio_decomposed)
                    z = encoder_outputs.pop()

                encoder_outputs = encoder_outputs[::-1]
                output_decomposed = decoder(z, encoder_outputs)
            else:
                # TCN architecture
                output_decomposed = encoder(dry_audio_decomposed)

            dry = pqmf.inverse(dry_audio_decomposed)
            output = pqmf.inverse(output_decomposed)
            wet = pqmf.inverse(wet_audio_decomposed)

            if decoder == None:
                rf = encoder.compute_receptive_field()
                print(rf)
                print(dry.shape)
                print(wet.shape)
                print(output.shape)
                wet = wet[..., rf:dry.shape[-1]]

            loss = criterion(output , wet)

            train_epoch_criterion += loss
            
            if use_kl:
                loss += kl_div

            train_epoch_loss += loss 

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({'epoch': f'{epoch + 1}/{num_epochs}', 'loss': f'{loss.item():.4f}'})

        scheduler.step()

        train_avg_epoch_loss = train_epoch_loss / len(train_loader)
        train_avg_epoch_loss_criterion = train_epoch_criterion / len(train_loader)  
        if use_kl:
            train_avg_epoch_kl_div = train_epoch_kl_div / len(train_loader)

        # Log loss
        tensorboard_writer.add_scalar("Loss/ training loss", train_avg_epoch_loss, epoch)
        tensorboard_writer.add_scalar("Loss/ training criterion", train_avg_epoch_loss_criterion, epoch)
        if use_kl:
            tensorboard_writer.add_scalar("Loss/training kl_div", train_avg_epoch_kl_div, epoch)
        if decoder:
            for i, alpha in enumerate(decoder.get_alpha_values()):
                tensorboard_writer.add_scalar(f"Alpha/Block_{i}", alpha, epoch)
        # Log audio samples
        tensorboard_writer.add_audio("Audio/TCN_Input", dry_audio[0].cpu(), epoch, sample_rate=sample_rate)
        tensorboard_writer.add_audio("Audio/TCN_Target", wet_audio[0].cpu(), epoch, sample_rate=sample_rate)
        tensorboard_writer.add_audio("Audio/TCN_output", output[0].cpu(), epoch, sample_rate=sample_rate)
        tensorboard_writer.step()

        

        # Validation loop
        encoder.eval()
        if decoder:
            decoder.eval()
        val_epoch_loss = 0
        val_epoch_kl_div = 0
        val_epoch_criterion = 0

        with torch.no_grad():
            for batch, (dry_audio, wet_audio) in enumerate(val_loader):
                dry_audio = dry_audio.to(device)
                wet_audio = wet_audio.to(device)

                # Pad both dry and wet audio to next power of 2
                dry_audio = center_pad_next_pow_2(dry_audio)
                wet_audio = center_pad_next_pow_2(wet_audio)

                # Apply PQMF to input
                dry_audio_decomposed = pqmf(dry_audio)
                wet_audio_decomposed = pqmf(wet_audio)

                # Throw error if wet audio is longer than dry audio
                if wet_audio_decomposed.shape[-1] != dry_audio_decomposed.shape[-1]:
                    raise ValueError(f"Wet audio is not the same length as dry audio: {wet_audio_decomposed.shape[-1]} vs {dry_audio_decomposed.shape[-1]}")

                 # Forward pass
                if decoder:
                    # Encoder-Decoder architecture
                    if use_kl:
                        mu, logvar, encoder_outputs = encoder(dry_audio_decomposed)
                        z = encoder.reparameterize(mu, logvar)
                        kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.shape[-1]
                        train_epoch_kl_div += kl_div
                    else:
                        encoder_outputs = encoder(dry_audio_decomposed)
                        z = encoder_outputs.pop()

                    encoder_outputs = encoder_outputs[::-1]
                    output_decomposed = decoder(z, encoder_outputs)
                else:
                    # TCN architecture
                    output_decomposed = encoder(dry_audio_decomposed)

                dry = pqmf.inverse(dry_audio_decomposed)
                output = pqmf.inverse(output_decomposed)
                wet = pqmf.inverse(wet_audio_decomposed)

                loss = criterion(output, wet)

                val_epoch_criterion += loss

                if use_kl:
                    loss += kl_div

                val_epoch_loss += loss

        val_avg_epoch_loss = val_epoch_loss / len(val_loader)
        val_avg_epoch_loss_criterion = val_epoch_criterion / len(val_loader)
        if use_kl:
            val_avg_epoch_kl_div = val_epoch_kl_div / len(val_loader)

        # Log loss
        tensorboard_writer.add_scalar("Loss/validation loss", val_avg_epoch_loss, epoch)
        tensorboard_writer.add_scalar("Loss/validation criterion", val_avg_epoch_loss_criterion, epoch)
        if use_kl:
            tensorboard_writer.add_scalar("Loss/validation kl_div", val_avg_epoch_kl_div, epoch)

        # Log audio samples (assuming you want to log validation audio as well)
        tensorboard_writer.add_audio("Audio/Val_Input", dry_audio[0].cpu(), epoch, sample_rate=sample_rate)
        tensorboard_writer.add_audio("Audio/Val_Target", wet_audio[0].cpu(), epoch, sample_rate=sample_rate)
        tensorboard_writer.add_audio("Audio/Val_Output", output[0].cpu(), epoch, sample_rate=sample_rate)

    progress_bar.close()
    tensorboard_writer.flush()

    print('Finished Training')
