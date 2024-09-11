import itertools
import subprocess
import os
import math

def calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size):
    # Step 1: Pad input to next power of 2
    if n_bands > 1:
        padded_input_size = 2 ** math.ceil(math.log2(input_size))
    else:
        padded_input_size = input_size
        
    # Step 2: Apply PQMF
    pqmf_size = padded_input_size // n_bands
    
    # Step 3: Calculate TCN receptive field
    rf = kernel_size - 1
    for i in range(1, n_blocks):
        dilation = dilation_growth ** i
        rf += (kernel_size - 1) * dilation
    
    # Step 4: Calculate final output size
    final_size = pqmf_size - rf
    
    return final_size

def generate_hyperparams():
    n_blocks = 4
    kernel_size = 13
    dilation_growth = 10
    
    n_blocks_range = range(max(1, base_n_blocks - 1), base_n_blocks + 1)
    kernel_size_range = range(max(3, base_kernel_size - 4), base_kernel_size + 5, 2)
    dilation_growth_range = range(max(2, base_dilation_growth - 2), base_dilation_growth + 2)
    
    # Add boolean options for the new parameters
    use_kl_options = [True, False]
    use_spectral_options = [True, False]
    use_adversarial_options = [True, False]
    use_skips_options = [True, False]

    for use_kl, use_spectral, use_adversarial, use_skips in itertools.product(
        use_kl_options, use_spectral_options, use_adversarial_options, use_skips_options
    ):
        yield n_blocks, kernel_size, dilation_growth, use_kl, use_spectral, use_adversarial, use_skips

def submit_batch_job(n_blocks, kernel_size, dilation_growth, use_kl, use_spectral, use_adversarial, use_skips):
    input_size = 508032  # From params.yaml
    n_bands = 1  # Fixed as per request
    
    final_size = calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size)
    
    if final_size <= 0:
        print(f"Skipping invalid configuration: n_blocks={n_blocks}, kernel_size={kernel_size}, dilation_growth={dilation_growth}")
        return
    
    env = {
        **os.environ,
        "EXP_PARAMS": (f"-S train.n_bands={n_bands} "
                       f"-S train.n_blocks={n_blocks} "
                       f"-S train.kernel_size={kernel_size} "
                       f"-S train.dilation_growth={dilation_growth} "
                       f"-S train.use_kl={str(use_kl).lower()} "
                       f"-S train.use_spectral={str(use_spectral).lower()} "
                       f"-S train.use_adversarial={str(use_adversarial).lower()} "
                       f"-S train.use_skip={str(use_skips).lower()}")
    }
    subprocess.run(['/usr/bin/bash', '-c', 'sbatch slurm_job.sh'], env=env)
    print(f"Submitted job: n_blocks={n_blocks}, kernel_size={kernel_size}, dilation_growth={dilation_growth}, "
          f"use_kl={use_kl}, use_spectral={use_spectral}, use_adversarial={use_adversarial}, use_skips={use_skips}")

if __name__ == "__main__":
    for params in generate_hyperparams():
        submit_batch_job(*params)