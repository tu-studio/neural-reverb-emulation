import math

def optimize_compression_rate(target_compression_rate, input_length, n_bands, n_blocks, kernel_size, dilation_growth, stride=2, dilate_conv=False):
    def calculate_compression(n_blocks, kernel_size, dilation_growth, paddings, stride=2,n_bands=2):
        output_length = input_length
        if n_bands > 1:
            padded_input_size = 2 ** math.ceil(math.log2(input_length))
        else:
            padded_input_size = input_length

        output_length = padded_input_size // n_bands
        
        print(f"Initial output length after n_bands: {output_length}")
        
        for i in range(n_blocks):
            dilation = dilation_growth ** (i + 1)
            padding = paddings[i] if i < len(paddings) else 0
            prev_output_length = output_length
            output_length = (output_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            print(f"Block {i}: dilation={dilation}, padding={padding}, output_length changed from {prev_output_length} to {output_length}")
            if output_length <= 0:
                return 0  # Invalid configuration

        # Final convolutional layer (conv_latent)
        final_dilation = dilation_growth ** n_blocks if dilate_conv else 1
        final_kernel_size = kernel_size
        final_padding = 2
        final_stride = 1
        prev_output_length = output_length
        output_length = ((output_length + 2 * final_padding - final_dilation * (final_kernel_size - 1) - 1) // final_stride) + 1
        print(f"Final conv layer: output_length changed from {prev_output_length} to {output_length}")

        if output_length <= 0:
            return 0  # Invalid configuration

        compression_rate = input_length / output_length
        print(f"Compression rate: {compression_rate:.2f}x")
        return compression_rate

    def find_optimal_parameters():
        best_params = None
        best_rate = 0
        
        for kernel_size_try in range(max(1, kernel_size - 5), kernel_size + 10, 2):
            for n_blocks_try in range(max(1, n_blocks - 5), n_blocks + 5):
                for dilation_growth_try in range(max(1, dilation_growth - 5), dilation_growth + 5):
                    for stride_try in range(1, 3):
                        rate = calculate_compression(n_blocks_try, kernel_size_try, dilation_growth_try, [0] * n_blocks_try, stride_try)
                        print(rate)
                        if abs(rate - target_compression_rate) <= abs(best_rate - target_compression_rate)  and rate >= target_compression_rate:
                            best_rate = rate
                            best_params = (n_blocks_try, kernel_size_try, dilation_growth_try, stride_try)
                            print(f"New best parameters found: n_blocks={n_blocks_try}, kernel_size={kernel_size_try}, dilation_growth={dilation_growth_try}")
                            print(f"New best rate: {best_rate:.2f}x")
                        
                        if best_rate >= target_compression_rate:
                            break
                    if best_rate >= target_compression_rate:
                            break
                if best_rate >= target_compression_rate:
                    break
            if best_rate >= target_compression_rate:
                break

        
        if best_params is None:
            raise ValueError("Could not find valid parameters to achieve target compression rate")
        
        return best_params

    def fine_tune_with_padding(n_blocks, kernel_size, dilation_growth):
        paddings = [0] * n_blocks
        best_paddings = paddings.copy()
        best_diff = float('inf')
        best_rate = calculate_compression(n_blocks, kernel_size, dilation_growth, paddings)

        for iteration in range(1000):  # Limit iterations to prevent infinite loops
            improved = False
            for i in range(n_blocks):
                for delta in [-1, 0, 1]:  # Small padding adjustments
                    new_paddings = paddings.copy()
                    new_paddings[i] += delta
                    if min(new_paddings) < 0:
                        continue

                    print(f"\nTrying paddings: {new_paddings}")
                    current_rate = calculate_compression(n_blocks, kernel_size, dilation_growth, new_paddings)
                    if current_rate == 0:  # Invalid configuration
                        continue
                    diff = abs(current_rate - target_compression_rate)

                    if diff < best_diff:
                        best_diff = diff
                        best_rate = current_rate
                        best_paddings = new_paddings.copy()
                        improved = True
                        print(f"Improvement found in iteration {iteration}, Block {i}, Delta {delta}")
                        print(f"New best paddings: {best_paddings}")
                        print(f"New best rate: {best_rate:.2f}x, Diff: {best_diff:.2f}")

            if not improved:
                print(f"No improvement in iteration {iteration}")
                break
            paddings = best_paddings.copy()

        return best_paddings, best_rate

    print("Phase 1: Finding optimal parameters")
    optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride = find_optimal_parameters()
    
    print("\nPhase 2: Fine-tuning with padding")
    optimal_paddings, achieved_rate = fine_tune_with_padding(optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth)
    
    return optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride, optimal_paddings, achieved_rate

# Example usage
input_length = 508032
n_bands = 2
n_blocks = 3
kernel_size = 17
dilation_growth = 15
target_compression_rate = 128

print("Initial configuration:")
print(f"input_length: {input_length}")
print(f"n_bands: {n_bands}")
print(f"n_blocks: {n_blocks}")
print(f"kernel_size: {kernel_size}")
print(f"dilation_growth: {dilation_growth}")
print(f"target_compression_rate: {target_compression_rate}")
print("\nStarting optimization...\n")

optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride,optimal_paddings, achieved_rate = optimize_compression_rate(
    target_compression_rate, input_length, n_bands, n_blocks, kernel_size, dilation_growth
)

print(f"\nFinal Results:")
print(f"Optimal n_blocks: {optimal_n_blocks}")
print(f"Optimal kernel_size: {optimal_kernel_size}")
print(f"Optimal dilation_growth: {optimal_dilation_growth}")
print(f"Optimal paddings: {optimal_paddings}")
print(f"Optimal stride: {optimal_stride}")
print(f"Achieved compression rate: {achieved_rate:.2f}x")
print(f"Target compression rate: {target_compression_rate:.2f}x")