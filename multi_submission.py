import itertools
import subprocess
import os
import math

def calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size):
    if n_bands > 1:
        padded_input_size = 2 ** math.ceil(math.log2(input_size))
    else:
        padded_input_size = input_size
    
    pqmf_size = padded_input_size // n_bands
    
    rf = 0
    for i in range(n_blocks):
        dilation = dilation_growth ** (i + 1)
        rf += (kernel_size - 1) * dilation
    
    final_size = pqmf_size - rf
    
    return final_size

def find_optimal_params(input_size, n_bands, n_blocks):
    max_kernel_size = 3
    max_dilation_growth = 1
    
    for kernel_size in range(3, 18):
        for dilation_growth in range(1, 16):
            final_size = calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size)
            if final_size > 0:
                max_kernel_size = kernel_size
                max_dilation_growth = dilation_growth
            else:
                return max_kernel_size, max_dilation_growth
    
    return max_kernel_size, max_dilation_growth

def generate_hyperparams():
    configurations = [
        (2, 17, 3, 15),
        (1, 13, 4, 10),
        (16, 3, 6, 4)
    ]
    
    use_kl_options = [False]
    use_adversarial_options = [False]
    use_skips_options = [True]
    use_noise_options = [False]
    use_residual_stack_options = [True, False]
    use_wn_options = [True, False]
    use_batch_norm_options = [True, False]
    loss_function_options = ['combined', 'mse', 'spectral_distance']

    for (n_bands, kernel_size, n_blocks, dilation_growth), use_kl, use_adversarial, use_skips, use_noise, use_residual_stack, use_wn, use_batch_norm, loss_function in itertools.product(
        configurations,
        use_kl_options, use_adversarial_options, use_skips_options, use_noise_options,
        use_residual_stack_options, use_wn_options, use_batch_norm_options,
        loss_function_options
    ):
        yield n_bands, kernel_size, n_blocks, dilation_growth, use_kl, use_adversarial, use_skips, use_noise, use_residual_stack, use_wn, use_batch_norm, loss_function

def submit_batch_job(n_bands, kernel_size, n_blocks, dilation_growth, use_kl, use_adversarial, use_skips, use_noise, use_residual_stack, use_wn, use_batch_norm, loss_function):
    env = {
        **os.environ,
        "EXP_PARAMS": (f"-S train.n_bands={n_bands} "
                       f"-S train.n_blocks={n_blocks} "
                       f"-S train.kernel_size={kernel_size} "
                       f"-S train.dilation_growth={dilation_growth} "
                       f"-S train.use_kl={str(use_kl).lower()} "
                       f"-S train.use_adversarial={str(use_adversarial).lower()} "
                       f"-S train.use_noise={str(use_noise).lower()} "
                       f"-S train.use_skip={str(use_skips).lower()} "
                       f"-S train.use_residual={str(use_residual_stack).lower()} "
                       f"-S train.use_wn={str(use_wn).lower()} "
                       f"-S train.use_batch_norm={str(use_batch_norm).lower()} "
                       f"-S metrics.loss_function={loss_function}")
    }
    subprocess.run(['/usr/bin/bash', '-c', 'sbatch slurm_job.sh'], env=env)
    print(f"Submitted job: n_bands={n_bands}, n_blocks={n_blocks}, kernel_size={kernel_size}, dilation_growth={dilation_growth}, "
          f"use_kl={use_kl}, use_adversarial={use_adversarial}, use_skips={use_skips}, use_noise={use_noise}, "
          f"use_residual={use_residual_stack}, use_wn={use_wn}, use_batch_norm={use_batch_norm}, "
          f"loss_function={loss_function}")

if __name__ == "__main__":
    total_configurations = 0
    for params in generate_hyperparams():
        submit_batch_job(*params)
        total_configurations += 1
    
    print(f"Total configurations submitted: {total_configurations}")