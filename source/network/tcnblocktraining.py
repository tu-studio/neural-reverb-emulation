import torch
from torch.nn.utils import clip_grad_norm_

def train_tcn_with_visualization(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, num_epochs, device, sample_rate):
    rf = model.compute_receptive_field()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch, (dry_audio, wet_audio) in enumerate(train_loader):
            dry_audio, wet_audio = dry_audio.to(device), wet_audio.to(device)
            optimizer.zero_grad()
            
            output, intermediate_outputs = model(dry_audio)
            
            wet_audio_trimmed = wet_audio[..., rf-1:]
            
            loss = criterion(output, wet_audio_trimmed)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log intermediate outputs

            for i, inter_output in enumerate(intermediate_outputs):
                inter_output_trimmed = inter_output[..., :wet_audio.shape[-1]]
                block_rf = model.compute_receptive_field() // (2 ** (model.n_blocks - i - 1))
                wet_audio_block_trimmed = wet_audio[..., block_rf-1:]
                writer.add_scalar(f"Loss/Block_{i}", criterion(inter_output_trimmed, wet_audio_block_trimmed), epoch * len(train_loader) + batch)
                writer.add_audio(f"Audio/Block_{i}_Output", inter_output_trimmed[0].cpu(), epoch * len(train_loader) + batch, sample_rate=sample_rate)
        
        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, (dry_audio, wet_audio) in enumerate(val_loader):
                dry_audio, wet_audio = dry_audio.to(device), wet_audio.to(device)
                output, intermediate_outputs = model(dry_audio)
                

                wet_audio_trimmed = wet_audio[..., rf-1:]
                
                loss = criterion(output, wet_audio_trimmed)
                val_loss += loss.item()
                
                # Log intermediate outputs for validation
                
                for i, inter_output in enumerate(intermediate_outputs):
                    inter_output_trimmed = inter_output[..., :wet_audio.shape[-1]]
                    block_rf = model.compute_receptive_field() // (2 ** (model.n_blocks - i - 1))
                    wet_audio_block_trimmed = wet_audio[..., block_rf-1:]
                    writer.add_scalar(f"Val_Loss/Block_{i}", criterion(inter_output_trimmed, wet_audio_block_trimmed), epoch * len(val_loader) + batch)
                    writer.add_audio(f"Val_Audio/Block_{i}_Output", inter_output_trimmed[0].cpu(), epoch * len(val_loader) + batch, sample_rate=sample_rate)
        
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        
        scheduler.step()
        
    print('Finished Training')