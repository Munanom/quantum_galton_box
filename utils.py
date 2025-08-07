import numpy as np
from scipy.stats import entropy
from typing import List, Dict

def compute_kl_divergence(experimental_probs: np.ndarray, theoretical_probs: np.ndarray) -> float:
    """Compute Kullback-Leibler divergence with small epsilon to avoid log(0)."""
    eps = 1e-10
    return entropy(experimental_probs + eps, theoretical_probs + eps)

def compute_js_divergence(experimental_probs: np.ndarray, theoretical_probs: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence."""
    m = 0.5 * (experimental_probs + theoretical_probs)
    return 0.5 * (compute_kl_divergence(experimental_probs, m) + compute_kl_divergence(theoretical_probs, m))

def compute_distance_metrics(counts: Dict[str, int], theoretical_probs: np.ndarray, total_shots: int, num_bins: int) -> dict:
    """Compute distance metrics between experimental and theoretical distributions."""
    # Convert counts to probabilities
    experimental_probs = np.zeros(num_bins)
    for bitstring, count in counts.items():
        position = int(bitstring, 2) if bitstring else 0
        if position < num_bins:
            experimental_probs[position] = count / total_shots
    
    kl = compute_kl_divergence(experimental_probs, theoretical_probs)
    js = compute_js_divergence(experimental_probs, theoretical_probs)
    tv = 0.5 * np.sum(np.abs(experimental_probs - theoretical_probs))
    
    return {'kl_divergence': kl, 'js_divergence': js, 'total_variation': tv}

def compute_stochastic_uncertainty(
    simulate_fn, num_runs: int, shots: int, theoretical_probs: np.ndarray, num_bins: int, **kwargs
) -> dict:
    """Compute distance metrics with stochastic uncertainty over multiple runs."""
    distances = {'kl_divergence': [], 'js_divergence': [], 'total_variation': []}
    
    for _ in range(num_runs):
        counts = simulate_fn(shots=shots, **kwargs)
        metrics = compute_distance_metrics(counts, theoretical_probs, shots, num_bins)
        for key in distances:
            distances[key].append(metrics[key])
    
    return {
        key: {'mean': np.mean(values), 'std': np.std(values)}
        for key, values in distances.items()
    }