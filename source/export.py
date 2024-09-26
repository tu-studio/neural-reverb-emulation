import torch
from pathlib import Path
from utils import config
from network.encoder import EncoderTCN
from network.decoder import DecoderTCN
from network.tcn import TCN
from network.CombinedModels import CombinedEncoderDecoder

def main():
    # Load the hyperparameters from the params yaml file into a Dictionary
    params = config.Params('params.yaml')

    # Load the parameters from the dictionary into variables
    n_inputs = params["train"]["n_inputs"]
    n_bands = params["train"]["n_bands"]
    latent_dim = params["train"]["latent_dim"]
    kernel_size = params["train"]["kernel_size"]
    n_blocks = params["train"]["n_blocks"]
    dilation_growth = params["train"]["dilation_growth"]
    n_channels = params["train"]["n_channels"]
    use_kl = params["train"]["use_kl"]
    input_size = params["general"]["input_size"]
    use_skip = params["train"]["use_skip"]
    use_tcn = params["train"]["use_tcn"]
    use_noise = params["train"]["use_noise"]
    use_wn = params["train"]["use_wn"]
    use_batch_norm = params["train"]["use_batch_norm"]
    use_residual = params["train"]["use_residual"]
    use_latent = params["train"]["use_latent"]
    dilate_conv = params["train"]["dilate_conv"]
    activation = params["train"]["activation"]

    # Define the model based on the architecture used in training
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
            use_latent=use_latent,
            dilate_conv=dilate_conv,
            activation=activation
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
            activation=activation
        )
        
        model = CombinedEncoderDecoder(encoder, decoder)

        # Load the model state
        encoder_path = Path('models/checkpoints/encoder.pth')
        decoder_path = Path('models/checkpoints/decoder.pth')
        encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

        # Prepare example input
        example = torch.randn(1, n_bands, int(2**math.ceil(math.log2(input_size))/n_bands))

    else:
        model = TCN(
            n_inputs=n_inputs,
            n_outputs=n_inputs,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_channels=n_channels,
            dilation_growth=dilation_growth
        )

        # Load the model state
        model_path = Path('models/checkpoints/tcn_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        # Prepare example input
        example = torch.rand(1, n_inputs, input_size)

    # Export the model
    output_file_path = Path('models/exports/model.onnx')
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(model, example, output_file_path, export_params=True, opset_version=17, do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Model exported to ONNX format.")

if __name__ == "__main__":
    main()