import torch
import torchinfo
from torch.utils.data import  random_split, DataLoader
from network.encoder import EncoderTCN
from network.decoder import DecoderTCN
from network.discriminator import Discriminator
from network.tcn import TCN
from network.training import train
from network.testing import test
from network.dataset import AudioDataset
from network.metrics import spectral_distance, single_stft_loss, fft_loss, CombinedLoss
from network.CombinedModels import CombinedEncoderDecoder
from network.latent import calculate_final_input_size
from utils import logs, config
from pathlib import Path
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    use_tcn = params["train"]["use_tcn"]
    loss_function = params["metrics"]["loss_function"]
    additional_mse = params["metrics"]["additional_mse"]
    additional_spec = params["metrics"]["additional_spec"]
    additional_stft = params["metrics"]["additional_stft"]
    additional_fft = params["metrics"]["additional_fft"]
    use_pqmf = params["train"]["use_pqmf"]
    use_adversarial = params["train"]["use_adversarial"]
    gan_loss = params["gan"]["loss_type"]
    use_noise = params["train"]["use_noise"]
    use_wn = params["train"]["use_wn"]
    use_batch_norm = params["train"]["use_batch_norm"]
    use_residual = params["train"]["use_residual"]
    activation = params["train"]["activation"]
    combined_spectral_weight = params["metrics"]["combined_spectral_weight"]
    combined_mse_weight = params["metrics"]["combined_mse_weight"]
    dilate_conv = params["train"]["dilate_conv"]
    stride = params["train"]["stride"]
    additional_metrics = [ additional_spec ,additional_stft, additional_fft, additional_mse]

    final_size = calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size)
    print("final size = ", final_size)

    # Create a SummaryWriter object to write the tensorboard logs
    tensorboard_path = logs.return_tensorboard_path()
    metrics = {'Loss/ training loss': None, 'Test Loss/Overall': None}
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics)

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(random_seed)
    # Prepare the requested device for training. Use cpu if the requested device is not available 
    device = config.prepare_device(device_request)

    if not use_pqmf:
        n_bands = 1

    # Create the discriminator with the new parameters
    discriminator = Discriminator(
        n_layers=params["discriminator"]["n_layers"],
        base_channels=params["discriminator"]["n_channels"],
        kernel_size=params["discriminator"]["kernel_size"],
        stride=params["discriminator"]["stride"],
        padding=params["discriminator"]["padding"],
    ).to(device)

    # Update the discriminator optimizer
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=params["discriminator"]["lr"],
        betas=(params["discriminator"]["beta1"], params["discriminator"]["beta2"])
    )

    # Build the model
    if not use_tcn:
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
            activation=activation,
            stride=stride,
            dilate_conv=dilate_conv)
        
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
            activation=activation,
            stride=stride,
            dilate_conv=dilate_conv)
        
        random_input = torch.randn(1, n_bands, int(2**math.ceil(math.log2(input_size))/n_bands))
        random_skips = []
        x = random_input

        if use_kl:
            mu, logvar, encoder_outputs = encoder(x)
            encoder_outputs.pop()
            z, _ = encoder.reparameterize(mu, logvar)
        else:
            encoder_outputs = encoder(x)
            z = encoder_outputs.pop()
        encoder_outputs = encoder_outputs[::-1]


        print("Encoder Architecture")
        print("Input shape: ", random_input.shape)
        torchinfo.summary(encoder, input_data=random_input, device=device)

        print("Decoder Architecture")
        print("Input shape: ", z.shape)
        print("Random skips shape: ", [i.shape for i in encoder_outputs])
        torchinfo.summary(decoder, input_data=[z, encoder_outputs], device=device)

        combined_model = CombinedEncoderDecoder(encoder, decoder)

        # Count and log total parameters
        total_params = count_parameters(combined_model)
        writer.add_scalar("Model/Total_Parameters", total_params, 0)
        print(f"Total trainable parameters: {total_params}")

        # Add the combined model graph to TensorBoard
        random_input = torch.randn(1, n_bands, int(2**math.ceil(math.log2(input_size))/n_bands))
        writer.add_graph(combined_model, random_input.to(device))

        # Setup optimizer
        model_params = list(encoder.parameters())
        model_params += list(decoder.parameters())
        optimizer = torch.optim.Adam(model_params, lr, (0.5, 0.9))
    
    elif use_tcn:
        model = TCN(
            n_inputs=n_inputs,
            n_outputs=n_inputs,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_channels=n_channels,
            dilation_growth=dilation_growth
        )

        # Count and log total parameters
        total_params = count_parameters(model)
        writer.add_scalar("Model/Total_Parameters", total_params, 0)
        print(f"Total trainable parameters: {total_params}")
        
        print("TCN Architecture")
        torchinfo.summary(model, input_data=torch.randn(1, n_inputs, input_size), device=device)

        # Add the model graph to the tensorboard logs
        writer.add_graph(model, torch.randn(1, n_inputs, input_size).to(device))

        model_params = list(model.parameters())

    # Use the appropriate loss function based on the parameter
    if loss_function == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_function == "spectral_distance":
        criterion = spectral_distance
    elif loss_function == "single_stft_loss":
        criterion = single_stft_loss
    elif loss_function == "fft_loss":
        criterion = fft_loss
    elif loss_function == "combined":
        criterion = CombinedLoss(mse_weight=combined_spectral_weight, spectral_weight=combined_spectral_weight)
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    # Setup optimizer
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

    # Modify the training and testing calls to work with both model types
    if not use_tcn:
        train(encoder, decoder, discriminator,train_loader, val_loader, criterion, optimizer, d_optimizer, scheduler,
              tensorboard_writer=writer, num_epochs=n_epochs, device=device,
              n_bands=n_bands, use_kl=use_kl, use_adversarial=use_adversarial, sample_rate=sample_rate, additional_metrics= additional_metrics, gan_loss = gan_loss)
        
        test(encoder, decoder, test_loader, criterion, writer, device, n_bands, use_kl, sample_rate)
        
        # Save the models
        save_path = Path('models/checkpoints')
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(encoder.state_dict(), save_path / 'encoder.pth')
        torch.save(decoder.state_dict(), save_path / 'decoder.pth')
    else:
        train(model, None, train_loader, val_loader, criterion, optimizer, scheduler,
              tensorboard_writer=writer, num_epochs=n_epochs, device=device,
              n_bands=n_bands, use_kl=False, sample_rate=sample_rate)
        
        test(model, None, test_loader, criterion, writer, device, n_bands, False, sample_rate)
        
        # Save the model
        save_path = Path('models/checkpoints')
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path / 'tcn_model.pth')

    writer.close()

    print("Done !")

if __name__ == "__main__":
    main()
