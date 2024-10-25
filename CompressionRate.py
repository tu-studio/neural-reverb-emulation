import math
import itertools
from typing import Dict, List

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
            return 0

    final_dilation = dilation_growth ** n_blocks if dilate_conv else 1
    final_kernel_size = kernel_size
    final_padding = 0
    final_stride = 1
    output_length = ((output_length + 2 * final_padding - final_dilation * (final_kernel_size - 1) - 1) // final_stride) + 1

    if output_length <= 0:
        return 0

    receptive_field = padded_input_size - output_length
    return receptive_field

def find_best_config_for_band(target_receptive_field: int, input_length: int, n_bands: int, stride: int = 2, max_blocks: int = 3) -> Dict:
    best_config = None
    best_difference = float('inf')
    
    # Restrict search space for fewer blocks
    n_blocks_range = range(1, max_blocks + 1)
    kernel_size_range = range(3, 20)
    dilation_growth_range = range(1, 16)

    for n_blocks, kernel_size, dilation_growth in itertools.product(n_blocks_range, kernel_size_range, dilation_growth_range):
        # First try without padding
        receptive_field = calculate_receptive(n_blocks, kernel_size, dilation_growth, 0, input_length, stride, n_bands)
        difference = abs(receptive_field - target_receptive_field)
        
        if receptive_field > 0 and difference < best_difference:
            best_difference = difference
            best_config = {
                'n_blocks': n_blocks,
                'kernel_size': kernel_size,
                'dilation_growth': dilation_growth,
                'n_bands': n_bands,
                'padding': 0,
                'receptive_field': receptive_field,
                'difference': difference
            }

    # Fine-tune with padding if we found a configuration
    if best_config:
        for padding in range(0, 100):  # Reduced padding range for efficiency
            receptive_field = calculate_receptive(
                best_config['n_blocks'],
                best_config['kernel_size'],
                best_config['dilation_growth'],
                padding,
                input_length,
                stride,
                n_bands
            )
            difference = abs(receptive_field - target_receptive_field)
            
            if difference < best_config['difference']:
                best_config.update({
                    'padding': padding,
                    'receptive_field': receptive_field,
                    'difference': difference
                })

    return best_config

def grid_search_all_bands(target_receptive_field: int, input_length: int, stride: int = 2, max_blocks: int = 3) -> Dict[int, Dict]:
    band_configs = {}
    band_values = [1, 2, 4, 8, 16]
    
    for n_bands in band_values:
        print(f"\nSearching configuration for {n_bands} bands...")
        config = find_best_config_for_band(target_receptive_field, input_length, n_bands, stride, max_blocks)
        if config:
            band_configs[n_bands] = config
            print(f"Found configuration for {n_bands} bands with {config['n_blocks']} blocks")
    
    return band_configs

def print_results(configurations: Dict[int, Dict]) -> None:
    print("\n=== Best Configurations for Each Number of Bands ===")
    print("\n{:<6} {:<8} {:<12} {:<16} {:<8} {:<15} {:<15}".format(
        "Bands", "Blocks", "Kernel Size", "Dilation Growth", "Padding", "Receptive Field", "Difference"
    ))
    print("-" * 80)
    
    for n_bands, config in configurations.items():
        print("{:<6} {:<8} {:<12} {:<16} {:<8} {:<15} {:<15}".format(
            n_bands,
            config['n_blocks'],
            config['kernel_size'],
            config['dilation_growth'],
            config['padding'],
            config['receptive_field'],
            config['difference']
        ))

# Example usage
if __name__ == "__main__":
    target_receptive_field = 2**17
    input_length = 2**19
    stride = 1
    max_blocks = 3  # Limiting to 3 blocks maximum

    configurations = grid_search_all_bands(target_receptive_field, input_length, stride, max_blocks)
    print_results(configurations)