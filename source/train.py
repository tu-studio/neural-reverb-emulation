import torch
import torchinfo
from torch.utils.data import  random_split, DataLoader
from network.encoder import EncoderTCN
from network.decoder import DecoderTCN
from network.training import train
from network.testing import test
from network.dataset import AudioDataset
from network.metrics import spectral_distance
from utils import logs, config
from pathlib import Path
import math

def main():
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params('params.yaml')

    n_inputs = params["train"]["n_inputs"]
    n_bands = params["train"]["n_bands"]
    latent_dim = params["train"]["latent_dim"]
    n_epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    kernel_size = params["train"]["kernel_size"]
    n_blocks = params["train"]["n_blocks"]
    dilation_growth = params["train"]["dilation_growth"]
    n_channels = params["train"]["n_channels"]
    lr = params["train"]["lr"]
    use_kl = params["train"]["use_kl"]
    device_request = params["train"]["device_request"]
    random_seed = params["general"]["random_seed"]
    input_file = params["train"]["input_file"]
    input_size = params["general"]["input_size"]
    scheduler_rate = params["train"]["scheduler_rate"]
    use_skip = params["train"]["use_skip"]

    # Create a SummaryWriter object to write the tensorboard logs
    tensorboard_path = logs.return_tensorboard_path()
    metrics = {}
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics)

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.prepare_device(device_request)

        # Build the model
    encoder = EncoderTCN(
        n_inputs=n_bands,
        kernel_size=kernel_size, 
        n_blocks=n_blocks, 
        dilation_growth=dilation_growth, 
        n_channels=n_channels,
        latent_dim=latent_dim,
        use_kl=use_kl)
    
    decoder = DecoderTCN(
        n_outputs=n_bands,
        kernel_size=kernel_size,
        n_blocks=n_blocks, 
        dilation_growth=dilation_growth, 
        n_channels=n_channels,
        latent_dim=latent_dim,
        use_kl=use_kl,
        use_skip=use_skip)
    
    random_input = torch.randn(1, n_bands, int(2**math.ceil(math.log2(input_size))/n_bands))
    random_skips = []
    x = random_input
    for block in encoder.blocks:
        x = block(x)
        random_skips.append(x)
    random_skips.pop()
    random_skips = random_skips[::-1]
    random_skips.append(random_input)

    print("Encoder Architecture")
    print("Input shape: ", random_input.shape)
    torchinfo.summary(encoder, input_data=random_input, device=device)

    print("Decoder Architecture")
    print("Input shape: ", x.shape)
    print("Random skips shape: ", [i.shape for i in random_skips])
    torchinfo.summary(decoder, input_data=[x, random_skips], device=device)

    # Add the model graphs to the tensorboard logs
    writer.add_graph(encoder, random_input.to(device))
    writer.add_graph(decoder, [x.to(device), [skip.to(device) for skip in random_skips]])

    # setup loss function, optimizer, and scheduler
    criterion = spectral_distance

    # Setup optimizer
    model_params = list(encoder.parameters())
    model_params += list(decoder.parameters())
    optimizer = torch.optim.Adam(model_params, lr, (0.5, 0.9))

    scheduler_milestones = [0.5,0.8,0.95]

    # Convert percentages to iteration numbers
    milestones = [int(n_epochs * p) for p in scheduler_milestones]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=scheduler_rate,
        verbose=False,
    )

    # Load the dataset
    full_dataset = AudioDataset(input_file)

    # Define the sizes of your splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Get the sample rate
    sample_rate = full_dataset.get_sample_rate()

    print(f"Full dataset size: {total_size}")
    print(f"Full dataset: {full_dataset}")

    # Create the splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(f"Train dataset: {train_dataset}")

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train(encoder, 
          decoder, 
          train_loader, 
          val_loader, 
          criterion, 
          optimizer, 
          scheduler,
          tensorboard_writer=writer, 
          num_epochs=n_epochs, 
          device=device,
          n_bands=n_bands, 
          use_kl=use_kl, 
          sample_rate=sample_rate)

    # Evaluate the model
    test(encoder, 
         decoder, 
         test_loader,
         criterion, 
         writer, 
         device, 
         n_bands, 
         use_kl, 
         sample_rate)

    writer.close()

    # Save the model
    save_path = Path('models/checkpoints')
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(encoder.state_dict(), save_path / 'encoder.pth')
    torch.save(decoder.state_dict(), save_path / 'decoder.pth')
    print(f"Saved PyTorch Model States to {save_path}")

    print("Done with the training stage!")

if __name__ == "__main__":
    main()
