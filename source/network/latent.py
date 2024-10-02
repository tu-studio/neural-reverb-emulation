import math

def calculate_final_input_size(input_size, n_bands, dilation_growth, n_blocks, kernel_size):
    # Step 1: Pad input to next power of 2
    if n_bands > 1:
        padded_input_size = 2 ** math.ceil(math.log2(input_size))
    else:
        padded_input_size = input_size


    print(input_size, padded_input_size)
        
    # Step 2: Apply PQMF
    pqmf_size = padded_input_size // n_bands
    
    # Step 3: Calculate TCN receptive field
    rf = 0

    for i in range(n_blocks):
        dilation = dilation_growth ** (i+1)
        rf += (kernel_size - 1) * dilation


    rf += (kernel_size - 1) * dilation_growth ** n_blocks
    
    # Step 4: Calculate final output size
    final_size = pqmf_size - rf
    
    return final_size

print(calculate_final_input_size(524288, 16, 15, 3, 6)) # Expected output: 512