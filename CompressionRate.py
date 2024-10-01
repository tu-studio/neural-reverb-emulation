import math
import itertools

def calculate_receptive(n_blocks, kernel_size, dilation_growth, padding, input_length, stride=2, n_bands=2, dilate_conv=True):
    output_length = input_length
    if n_bands > 1:
        padded_input_size = 2 ** math.ceil(math.log2(input_length))
    else:
        padded_input_size = input_length

    output_length = padded_input_size // n_bands
    
    for i in range(n_blocks):
        dilation = dilation_growth ** (i + 1)
        output_length = (output_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        if output_length <= 0:
            return 0  # Invalid configuration

    # Final convolutional layer (conv_latent)
    final_dilation = dilation_growth ** n_blocks if dilate_conv else 1
    final_kernel_size = kernel_size
    final_padding = 0
    final_stride = 1
    output_length = ((output_length + 2 * final_padding - final_dilation * (final_kernel_size - 1) - 1) // final_stride) + 1

    if output_length <= 0:
        return 0  # Invalid configuration

    receptive_field = padded_input_size - output_length
    return receptive_field

def grid_search_receptive_field(target_receptive_field, input_length, stride=2, tolerance=1):
    best_config = None
    best_difference = float('inf')

    # Stage 1: Gross approach without padding
    n_blocks_range = range(1, 6)
    kernel_size_range = range(3, 20)
    dilation_growth_range = range(1, 16)
    n_bands_range = [2, 4, 8, 16]

    print("Stage 1: Gross approach without padding")
    for n_blocks, kernel_size, dilation_growth, n_bands in itertools.product(n_blocks_range, kernel_size_range, dilation_growth_range, n_bands_range):
        receptive_field = calculate_receptive(n_blocks, kernel_size, dilation_growth, 0, input_length, stride, n_bands)
        difference = abs(receptive_field - target_receptive_field)
        
        if difference < best_difference:
            best_difference = difference
            best_config = {
                'n_blocks': n_blocks,
                'kernel_size': kernel_size,
                'dilation_growth': dilation_growth,
                'n_bands': n_bands,
                'padding': 0,
                'receptive_field': receptive_field
            }
        
        if difference <= tolerance:
            return best_config

    # Stage 2: Fine-tuning with even padding
    print("Stage 2: Fine-tuning with even padding")
    if best_config:
        padding_range = range(0, 1000)  # Adjust this range as needed
        for padding in padding_range:
            receptive_field = calculate_receptive(
                best_config['n_blocks'],
                best_config['kernel_size'],
                best_config['dilation_growth'],
                padding,
                input_length,
                stride,
                best_config['n_bands']
            )
            difference = abs(receptive_field - target_receptive_field)
            
            if difference < best_difference:
                best_difference = difference
                best_config['padding'] = padding
                best_config['receptive_field'] = receptive_field
            
            if difference <= tolerance:
                break

    return best_config

# Example usage
target_receptive_field = 2**17
input_length = 2**19
stride = 1

result = grid_search_receptive_field(target_receptive_field, input_length, stride)

print(f"\nBest configuration found:")
print(f"n_blocks: {result['n_blocks']}")
print(f"kernel_size: {result['kernel_size']}")
print(f"dilation_growth: {result['dilation_growth']}")
print(f"n_bands: {result['n_bands']}")
print(f"padding: {result['padding']}")
print(f"Achieved receptive field: {result['receptive_field']}")
print(f"Difference from target: {abs(result['receptive_field'] - target_receptive_field)}")