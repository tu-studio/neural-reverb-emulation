import numpy as np

def get_beta_kl_cyclic_annealed(step, cycle_size, warmup, min_beta, max_beta):
    if step < warmup:
        return 0.
    step = step - warmup
    cycle = np.floor(step / cycle_size)
    x = step - cycle * cycle_size
    x = x / cycle_size
    return min_beta + (max_beta - min_beta) * 0.5 * (1 - np.cos(2 * np.pi * x))