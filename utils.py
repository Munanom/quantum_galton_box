import numpy as np
from typing import Dict, List
from scipy.stats import entropy

def compute_distance_metrics(counts: Dict[str, int], theoretical_probs: np.ndarray, total_shots: int, num_bins: int) -> Dict[str, float]:
    """
    Compute distance metrics between experimental and theoretical distributions.
    
    Args:
        counts: Dictionary of measurement outcomes and their counts
        theoretical_probs: Array of theoretical probabilities
        total_shots: Total number of shots
        num_bins: Number of possible outcomes (e.g., 9 for 0 to 8)
    
    Returns:
        Dictionary containing KL divergence, JS divergence, and total variation distance
    """
    experimental_probs = np.zeros(num_bins)
    
    for bitstring, count in counts.items():
        # Remove spaces and take only the first num_levels bits
        clean_bitstring = bitstring.replace(" ", "")[:num_bins-1]  # Use num_bins-1 to match number of qubits
        # Count number of 1s for position (binomial distribution)
        position = clean_bitstring.count('1')
        if position < num_bins:
            experimental_probs[position] += count / total_shots
    
    # Ensure probabilities sum to 1
    if experimental_probs.sum() > 0:
        experimental_probs = experimental_probs / experimental_probs.sum()
    
    # Compute KL divergence
    kl_div = entropy(experimental_probs, theoretical_probs, base=2) if experimental_probs.sum() > 0 else float('inf')
    
    # Compute Jensen-Shannon divergence
    m = 0.5 * (experimental_probs + theoretical_probs)
    js_div = 0.5 * (entropy(experimental_probs, m, base=2) + entropy(theoretical_probs, m, base=2)) if experimental_probs.sum() > 0 else float('inf')
    
    # Compute total variation distance
    tv_dist = 0.5 * np.sum(np.abs(experimental_probs - theoretical_probs))
    
    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'total_variation': tv_dist
    }

def compute_stochastic_uncertainty(
    simulation_func, num_runs: int, shots: int, theoretical_probs: np.ndarray, num_bins: int, **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute stochastic uncertainty in distance metrics.
    
    Args:
        simulation_func: Function to simulate the circuit
        num_runs: Number of runs to estimate uncertainty
        shots: Number of shots per simulation
        theoretical_probs: Theoretical probability distribution
        num_bins: Number of possible outcomes
        **kwargs: Additional arguments for simulation_func
    
    Returns:
        Dictionary with mean and std of distance metrics
    """
    kl_divs = []
    js_divs = []
    tv_dists = []
    
    for _ in range(num_runs):
        counts = simulation_func(shots=shots, **kwargs)
        metrics = compute_distance_metrics(counts, theoretical_probs, shots, num_bins)
        kl_divs.append(metrics['kl_divergence'])
        js_divs.append(metrics['js_divergence'])
        tv_dists.append(metrics['total_variation'])
    
    return {
        'kl_divergence': {'mean': np.mean(kl_divs), 'std': np.std(kl_divs)},
        'js_divergence': {'mean': np.mean(js_divs), 'std': np.std(js_divs)},
        'total_variation': {'mean': np.mean(tv_dists), 'std': np.std(tv_dists)}
    }