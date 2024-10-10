import itertools
import subprocess
import os
import math

def calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size, dilate_conv, dense=False):
    output_length = input_size
    stride=1
    if n_bands > 1:
        padded_input_size = 2 ** math.ceil(math.log2(input_size))
    else:
        padded_input_size = input_size

    output_length = padded_input_size // n_bands
    
    for _ in range(n_blocks):
        dilation = dilation_growth ** (_ + 1)
        # padding = (kernel_size - 1) * dilation  // 2
        padding = 0
        output_length = (output_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    if not dense:
        # Final convolutional layer (conv_latent)
        final_dilation = dilation_growth ** n_blocks
        final_kernel_size = kernel_size
        final_padding = 0
        final_stride = 1
        if dilate_conv:
            output_length = ((output_length + 2 * final_padding - final_dilation * (final_kernel_size - 1) - 1) // final_stride) + 1
        else:
            output_length = ((output_length + 2 * final_padding - (final_kernel_size - 1) - 1) // final_stride) + 1
    
    
    
    return output_length

def find_optimal_params(input_size, n_bands, n_blocks, dilate_conv=False, dense=False):
    max_kernel_size = 3
    max_dilation_growth = 1
    
    for dilation_growth in range(1, 16):
        for kernel_size in range(3, 18):
            final_size = calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size, dilate_conv, dense)
            if final_size > 0:
                max_kernel_size = kernel_size
                max_dilation_growth = dilation_growth
            else:
                return max_kernel_size, max_dilation_growth
    
    return max_kernel_size, max_dilation_growth

def generate_hyperparams(input_size):
    n_bands_options = [1]
    n_blocks_options = [4]
    latent_dim_options = [64]
    use_skips_options = [True, False]
    use_latent_options = ['conv']
    
    configurations = []
    
    # for n_bands, n_blocks in itertools.product(n_bands_options, n_blocks_options):
    #     kernel_size, dilation_growth = find_optimal_params(input_size, n_bands, n_blocks, True, False)
    #     configurations.append((n_bands, kernel_size, n_blocks, dilation_growth))

    configurations.append((1,16,4,8))
    
    use_kl_options = [False]
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
    input_size = 508032  # From params.yaml
    total_configurations = 0
    for params in generate_hyperparams(input_size):
        submit_batch_job(*params)
        total_configurations += 1
    
    print(f"Total configurations submitted: {total_configurations}")