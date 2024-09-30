import math

def optimize_receptive_field(target_receptive_field, input_length, n_blocks, kernel_size, dilation_growth, stride=2, dilate_conv=False, pqmf_kernel_size=8):
    def calculate_receptive_field(n_blocks, kernel_size, dilation_growth, paddings, stride=2):
        receptive_field = 1
        for i in range(n_blocks):
            dilation = dilation_growth ** i
            receptive_field += (kernel_size - 1) * dilation * stride + paddings[i] * 2

        # Final convolutional layer (conv_latent)
        final_dilation = dilation_growth ** n_blocks if dilate_conv else 1
        final_kernel_size = kernel_size
        receptive_field += (final_kernel_size - 1) * final_dilation

        # Account for PQMF
        receptive_field *= pqmf_kernel_size

        return receptive_field

    def find_optimal_parameters():
        best_params = None
        best_rf = 0
        
        for kernel_size_try in range(max(3, kernel_size - 5), kernel_size + 10, 2):
            for n_blocks_try in range(max(1, n_blocks - 5), n_blocks + 5):
                for dilation_growth_try in range(max(1, dilation_growth - 5), dilation_growth + 5):
                    for stride_try in range(1, 3):
                        rf = calculate_receptive_field(n_blocks_try, kernel_size_try, dilation_growth_try, [0] * n_blocks_try, stride_try)
                        if abs(rf - target_receptive_field) <= abs(best_rf - target_receptive_field):
                            best_rf = rf
                            best_params = (n_blocks_try, kernel_size_try, dilation_growth_try, stride_try)
                        
                        if best_rf == target_receptive_field:
                            return best_params, best_rf
        
        return best_params, best_rf

    def fine_tune_with_padding(n_blocks, kernel_size, dilation_growth, stride):
        best_paddings = [0] * n_blocks
        best_rf = calculate_receptive_field(n_blocks, kernel_size, dilation_growth, best_paddings, stride)
        
        for _ in range(100):  # Limit iterations to avoid infinite loop
            improved = False
            for i in range(n_blocks):
                for delta in [-1, 1]:
                    new_paddings = best_paddings.copy()
                    new_paddings[i] += delta
                    if min(new_paddings) >= 0:  # Ensure non-negative padding
                        new_rf = calculate_receptive_field(n_blocks, kernel_size, dilation_growth, new_paddings, stride)
                        if abs(new_rf - target_receptive_field) < abs(best_rf - target_receptive_field):
                            best_paddings = new_paddings
                            best_rf = new_rf
                            improved = True
            
            if not improved:
                break
        
        return best_paddings, best_rf

    print("\nStarting optimization...")
    optimal_params, achieved_rf = find_optimal_parameters()
    
    if optimal_params is None:
        print("Could not find valid parameters")
        return None
    
    optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride = optimal_params
    
    print("\nFine-tuning with padding...")
    optimal_paddings, fine_tuned_rf = fine_tune_with_padding(optimal_n_blocks, optimal_kernel_size, optimal_dilation_growth, optimal_stride)
    
    results = {
        "n_blocks": optimal_n_blocks,
        "kernel_size": optimal_kernel_size,
        "dilation_growth": optimal_dilation_growth,
        "stride": optimal_stride,
        "paddings": optimal_paddings,
        "achieved_rf": fine_tuned_rf
    }
    
    print(f"Achieved receptive field after fine-tuning: {fine_tuned_rf}")

    return results

# Example usage
input_length = 508032
n_blocks = 3
kernel_size = 17
dilation_growth = 15
target_receptive_field = 2**16  # 65536
pqmf_kernel_size = 8

print("Initial configuration:")
print(f"input_length: {input_length}")
print(f"n_blocks: {n_blocks}")
print(f"kernel_size: {kernel_size}")
print(f"dilation_growth: {dilation_growth}")
print(f"target_receptive_field: {target_receptive_field}")
print(f"pqmf_kernel_size: {pqmf_kernel_size}")

results = optimize_receptive_field(target_receptive_field, input_length, n_blocks, kernel_size, dilation_growth, pqmf_kernel_size=pqmf_kernel_size)

if results:
    print("\nFinal Results:")
    print(f"Optimal n_blocks: {results['n_blocks']}")
    print(f"Optimal kernel_size: {results['kernel_size']}")
    print(f"Optimal dilation_growth: {results['dilation_growth']}")
    print(f"Optimal stride: {results['stride']}")
    print(f"Optimal paddings: {results['paddings']}")
    print(f"Achieved receptive field: {results['achieved_rf']}")

print(f"\nTarget receptive field: {target_receptive_field}")