import itertools
import subprocess
import os
import math

def calculate_receptive_field(n_blocks, kernel_size, dilation_growth, input_length, n_bands, dilate_conv=True):
    if n_bands > 1:
        padded_input_size = 2 ** math.ceil(math.log2(input_length))
    else:
        padded_input_size = input_length
        
    output_length = padded_input_size // n_bands
    padding = 0
    stride = 1
    
    for i in range(n_blocks):
        dilation = dilation_growth ** (i + 1)
        output_length = (output_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        if output_length <= 0:
            return float('inf')  # Invalid configuration

    # Final convolutional layer
    final_dilation = dilation_growth ** n_blocks if dilate_conv else 1
    output_length = ((output_length + 2 * padding - final_dilation * (kernel_size - 1) - 1) // stride) + 1

    if output_length <= 0:
        return float('inf')

    receptive_field = padded_input_size - output_length
    return receptive_field

def find_optimal_params(input_size, n_bands):
    best_params = None
    best_receptive = 0
    
    # Search ranges
    n_blocks_range = range(2, 4)
    kernel_size_range = range(3, 17, 2)  # Odd numbers for kernel size
    dilation_growth_range = range(5, 12)
    
    for n_blocks, kernel_size, dilation_growth in itertools.product(
        n_blocks_range, kernel_size_range, dilation_growth_range
    ):
        receptive = calculate_receptive_field(
            n_blocks, kernel_size, dilation_growth, input_size, n_bands
        )
        
        # Check if receptive field is valid and maximized while being smaller than input_size
        if receptive < input_size and receptive > best_receptive:
            best_receptive = receptive
            best_params = (n_blocks, kernel_size, dilation_growth)
    
    return best_params, best_receptive

def generate_hyperparams(input_size):
    n_bands_options = [1]
    latent_dim_options = [64,128,32]
    use_skips_options = [True, False]
    use_latent_options = ['conv']
    
    configurations = []
    
    # Find optimal parameters for each number of bands
    for n_bands in n_bands_options:
        params = (4,16,8)
        if params is not None:
            n_blocks, kernel_size, dilation_growth = params
            configurations.append((n_bands, kernel_size, n_blocks, dilation_growth))
            print(f"For n_bands={n_bands}: blocks={n_blocks}, kernel={kernel_size}, "
                  f"dilation_growth={dilation_growth}")
    
    use_kl_options = [True, False]
    use_adversarial_options = [False]
    use_noise_options = [False]
    use_residual_stack_options = [True]
    use_wn_options = [False]
    use_batch_norm_options = [False]
    loss_function_options = ['combined']
    activation_options = ['prelu']

    for (n_bands, kernel_size, n_blocks, dilation_growth), latent_dim, use_kl, use_adversarial, use_skips, use_noise, use_residual_stack, use_wn, use_batch_norm, loss_function, activation, use_latent in itertools.product(
        configurations, latent_dim_options, use_kl_options, use_adversarial_options, use_skips_options, use_noise_options,
        use_residual_stack_options, use_wn_options, use_batch_norm_options,
        loss_function_options, activation_options, use_latent_options
    ):
        yield n_bands, kernel_size, n_blocks, dilation_growth, latent_dim, use_kl, use_adversarial, use_skips, use_noise, use_residual_stack, use_wn, use_batch_norm, loss_function, activation, use_latent

def submit_batch_job(n_bands, kernel_size, n_blocks, dilation_growth, latent_dim, use_kl, use_adversarial, use_skips, use_noise, use_residual_stack, use_wn, use_batch_norm, loss_function, activation, use_latent):
    env = {
        **os.environ,
        "EXP_PARAMS": (f"-S train.n_bands={n_bands} "
                       f"-S train.n_blocks={n_blocks} "
                       f"-S train.kernel_size={kernel_size} "
                       f"-S train.dilation_growth={dilation_growth} "
                       f"-S train.latent_dim={latent_dim} "
                       f"-S train.use_kl={str(use_kl).lower()} "
                       f"-S train.use_adversarial={str(use_adversarial).lower()} "
                       f"-S train.use_noise={str(use_noise).lower()} "
                       f"-S train.use_skip={str(use_skips).lower()} "
                       f"-S train.use_residual={str(use_residual_stack).lower()} "
                       f"-S train.use_wn={str(use_wn).lower()} "
                       f"-S train.use_batch_norm={str(use_batch_norm).lower()} "
                       f"-S metrics.loss_function={loss_function} "
                       f"-S train.activation={activation} "
                       f"-S train.use_latent={use_latent}")
    }
    subprocess.run(['/usr/bin/bash', '-c', 'sbatch slurm_job.sh'], env=env)
    print(f"Submitted job: n_bands={n_bands}, n_blocks={n_blocks}, kernel_size={kernel_size}, dilation_growth={dilation_growth}, "
          f"latent_dim={latent_dim}, use_kl={use_kl}, use_adversarial={use_adversarial}, use_skips={use_skips}, use_noise={use_noise}, "
          f"use_residual={use_residual_stack}, use_wn={use_wn}, use_batch_norm={use_batch_norm}, "
          f"loss_function={loss_function}, activation={activation}, use_latent={use_latent}")

if __name__ == "__main__":
    input_size =  524288 # From params.yaml
    total_configurations = 0
    for params in generate_hyperparams(input_size):
        submit_batch_job(*params)
        total_configurations += 1
    
    print(f"Total configurations submitted: {total_configurations}")