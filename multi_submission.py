import itertools
import subprocess
import os 

def calculate_hyperparams(n_bands):
    # Base values for n_bands = 1
    base_n_blocks = 4
    base_kernel_size = 13
    base_dilation_growth = 10
    
    # Scale hyperparameters based on n_bands
    n_blocks = max(2, base_n_blocks - (n_bands - 1) // 2)  # Decrease n_blocks as n_bands increases
    kernel_size = max(3, base_kernel_size - (n_bands - 1))  # Decrease kernel_size as n_bands increases
    dilation_growth = max(2, base_dilation_growth - (n_bands - 1))  # Decrease dilation_growth as n_bands increases
    
    return n_blocks, kernel_size, dilation_growth

def submit_batch_job(n_bands, use_spectral_loss):
    n_blocks, kernel_size, dilation_growth = calculate_hyperparams(n_bands)
    
    env = {
        **os.environ,
        "EXP_PARAMS": (f"-S train.n_bands={n_bands} "
                       f"-S train.n_blocks={n_blocks} "
                       f"-S train.kernel_size={kernel_size} "
                       f"-S train.dilation_growth={dilation_growth} "
                       f"-S train.use_spectral_loss={str(use_spectral_loss).lower()}")
    }
    subprocess.run(['/usr/bin/bash', '-c', 'sbatch slurm_job.sh'], env=env)

if __name__ == "__main__":
    n_bands_list = [1, 4, 8, 16]
    use_spectral_loss_list = [True, False]
    
    for n_bands, use_spectral_loss in itertools.product(n_bands_list, use_spectral_loss_list):
        submit_batch_job(n_bands, use_spectral_loss)