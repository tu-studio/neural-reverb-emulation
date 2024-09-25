import math

def optimize_compression_rate(target_compression_rate, input_length, n_blocks, kernel_size, dilation_growth, stride=2, dilate_conv=False):
    def calculate_compression(n_blocks, kernel_size, dilation_growth, paddings, stride=2, n_bands=2):
        output_length = input_length
        if n_bands > 1:
            padded_input_size = 2 ** math.ceil(math.log2(input_length))
        else:
            padded_input_size = input_length

        output_length = padded_input_size // n_bands
        
        for i in range(n_blocks):
            dilation = dilation_growth ** (i + 1)
            padding = paddings[i] if i < len(paddings) else 0
            output_length = (output_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            if output_length <= 0:
                return 0  # Invalid configuration

        # Final convolutional layer (conv_latent)
        final_dilation = dilation_growth ** n_blocks if dilate_conv else 1
        final_kernel_size = kernel_size
        final_padding = 2
        final_stride = 1
        output_length = ((output_length + 2 * final_padding - final_dilation * (final_kernel_size - 1) - 1) // final_stride) + 1

        if output_length <= 0:
            return 0  # Invalid configuration

        compression_rate = input_length / output_length
        return compression_rate

    def find_optimal_parameters(n_bands):
        best_params = None
        best_rate = 0
        
        for kernel_size_try in range(max(1, kernel_size - 5), kernel_size + 10, 2):
            for n_blocks_try in range(max(1, n_blocks - 5), n_blocks + 5):
                for dilation_growth_try in range(max(1, dilation_growth - 5), dilation_growth + 5):
                    for stride_try in range(1, 3):
                        rate = calculate_compression(n_blocks_try, kernel_size_try, dilation_growth_try, [0] * n_blocks_try, stride_try, n_bands)
                        if abs(rate - target_compression_rate) <= abs(best_rate - target_compression_rate) and rate >= target_compression_rate:
                            best_rate = rate
                            best_params = (n_blocks_try, kernel_size_try, dilation_growth_try, stride_try)
                        
                        if best_rate >= target_compression_rate:
                            return best_params, best_rate
        
        return best_params, best_rate

    def fine_tune_with_padding(n_bands, n_blocks, kernel_size, dilation_growth, stride):
        paddings = [0] * n_blocks
        best_paddings = paddings.copy()
        best_diff = float('inf')
        best_rate = calculate_compression(n_blocks, kernel_size, dilation_growth, paddings, stride, n_bands)

        # Phase 1: Add padding to all layers until we exceed the target rate
        while best_rate > target_compression_rate:
            new_paddings = [p + 1 for p in paddings]
            current_rate = calculate_compression(n_blocks, kernel_size, dilation_growth, new_paddings, stride, n_bands)
            
            if current_rate == 0:  # Invalid configuration
                break
            
            best_rate = current_rate
            best_paddings = new_paddings.copy()
            paddings = new_paddings

        # Phase 2: Fine-tune by adding padding to one layer at a time
        if best_rate < target_compression_rate:
            for iteration in range(1000):  # Limit iterations to prevent infinite loops
                improved = False
                for i in range(n_blocks):
                    new_paddings = best_paddings.copy()
                    new_paddings[i] -= 1
                    
                    current_rate = calculate_compression(n_blocks, kernel_size, dilation_growth, new_paddings, stride, n_bands)
                    if current_rate == 0:  # Invalid configuration
                        continue
                    
                    new_diff = abs(current_rate - target_compression_rate)
                    
                    if new_diff < best_diff:
                        best_diff = new_diff
                        best_rate = current_rate
                        best_paddings = new_paddings.copy()
                        improved = True
                        break  # Stop after improving one layer
                
                if not improved:
                    break  # Stop if no improvement was made in this iteration

        return best_paddings, best_rate

    results = {}
    for n_bands in [1, 2, 4, 8, 16]:
        print(f"\nOptimizing for n_bands = {n_bands}")
        optimal_params, initial_rate = find_optimal_parameters(n_bands)
        
        if optimal_params is None:
            print(f"Could not find valid parameters for n_bands = {n_bands}")
            continue
        
        optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride = optimal_params
        
        print("Fine-tuning with padding")
        optimal_paddings, achieved_rate = fine_tune_with_padding(n_bands, optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride)
        
        results[n_bands] = {
            "n_blocks": optimal_n_blocks,
            "kernel_size": optimal_kernel_size,
            "dilation_growth": optimal_dilation_growth,
            "stride": optimal_stride,
            "paddings": optimal_paddings,
            "achieved_rate": achieved_rate
        }
        
        print(f"Achieved compression rate: {achieved_rate:.2f}x")

    return results

# Example usage
input_length = 508032
n_blocks = 3
kernel_size = 17
dilation_growth = 15
target_compression_rate = 128

print("Initial configuration:")
print(f"input_length: {input_length}")
print(f"n_blocks: {n_blocks}")
print(f"kernel_size: {kernel_size}")
print(f"dilation_growth: {dilation_growth}")
print(f"target_compression_rate: {target_compression_rate}")
print("\nStarting optimization...\n")

results = optimize_compression_rate(target_compression_rate, input_length, n_blocks, kernel_size, dilation_growth)

print("\nFinal Results:")
for n_bands, result in results.items():
    print(f"\nn_bands: {n_bands}")
    print(f"Optimal n_blocks: {result['n_blocks']}")
    print(f"Optimal kernel_size: {result['kernel_size']}")
    print(f"Optimal dilation_growth: {result['dilation_growth']}")
    print(f"Optimal stride: {result['stride']}")
    print(f"Optimal paddings: {result['paddings']}")
    print(f"Achieved compression rate: {result['achieved_rate']:.2f}x")

print(f"\nTarget compression rate: {target_compression_rate:.2f}x")