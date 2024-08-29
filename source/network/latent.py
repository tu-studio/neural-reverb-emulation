import math
from utils import logs, config

def calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size):
    # Step 1: Pad input to next power of 2
    padded_input_size = 2 ** math.ceil(math.log2(input_size))
    
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

