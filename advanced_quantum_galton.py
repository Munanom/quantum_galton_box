#!/usr/bin/env python3
"""
Implements an advanced Quantum Galton Board based on the 'Universal Statistical Simulator' by Carney and Varcoe. The AdvancedQuantumGaltonBoard class:
- Simulates statistical distributions (binomial, exponential, Hadamard quantum walk) using quantum circuits.
- Creates a universal quantum circuit with rotation and entanglement operations to mimic a Galton Board.
- Simulates distributions with measurement and statevector analysis.
- Provides comprehensive visualization including experimental histograms, theoretical comparisons, error analysis, statistics, Gaussian convergence, noise optimization, and uncertainty analysis.
- Supports command-line arguments for customizing levels, bias, shots, distribution types, and additional analyses.
- Includes a demonstration mode to showcase all supported distributions.
The module extends the basic Galton Board simulation with flexible distribution types and detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2
from qiskit.result import LocalReadoutMitigator
from qiskit.quantum_info import Statevector
import seaborn as sns
from typing import List, Tuple, Dict
import math
from scipy.stats import binom, expon, norm
import argparse
from datetime import datetime
import warnings

# Placeholder implementations for utils functions
def compute_distance_metrics(counts: Dict[str, int], theoretical_probs: List[float], total_shots: int, num_states: int) -> Dict:
    """Compute distance metrics (KL, JS, TV) between experimental and theoretical distributions."""
    experimental_probs = np.zeros(num_states)
    for bitstring, count in counts.items():
        try:
            pos = int(bitstring.split()[0], 2) if ' ' in bitstring else int(bitstring, 2) if bitstring else 0
            if pos < num_states:
                experimental_probs[pos] = count / total_shots
        except ValueError as e:
            warnings.warn(f"Skipping invalid bitstring '{bitstring}': {e}")
    kl_div = sum(p * np.log(p / q + 1e-10) for p, q in zip(experimental_probs, theoretical_probs) if p > 0)
    js_div = 0.5 * (sum(p * np.log(p / ((p + q) / 2) + 1e-10) for p, q in zip(experimental_probs, theoretical_probs) if p > 0) +
                    sum(q * np.log(q / ((p + q) / 2) + 1e-10) for p, q in zip(experimental_probs, theoretical_probs) if q > 0))
    tv_dist = 0.5 * sum(abs(p - q) for p, q in zip(experimental_probs, theoretical_probs))
    return {'kl_divergence': kl_div, 'js_divergence': js_div, 'total_variation': tv_dist}

def compute_stochastic_uncertainty(simulate_fn, num_runs: int, shots: int, theoretical_probs: List[float], num_states: int, apply_mitigation: bool = False, **kwargs) -> Dict:
    """Compute uncertainty in distance metrics over multiple runs."""
    metrics = {'kl_divergence': [], 'js_divergence': [], 'total_variation': []}
    for _ in range(num_runs):
        counts = simulate_fn(shots=shots, apply_mitigation=apply_mitigation, **kwargs)
        result = compute_distance_metrics(counts, theoretical_probs, shots, num_states)
        for key in metrics:
            metrics[key].append(result[key])
    return {key: {'mean': np.mean(vals), 'std': np.std(vals)} for key, vals in metrics.items()}

class AdvancedQuantumGaltonBoard:
    """Advanced Quantum Galton Board with noise model and multiple distributions."""
    
    def __init__(self, num_levels: int = 8, bias: float = 0.5, use_noise: bool = False):
        if num_levels < 1:
            raise ValueError("num_levels must be positive")
        if not 0 <= bias <= 1:
            raise ValueError("bias must be between 0 and 1")
        self.num_levels = num_levels
        self.bias = bias
        self.num_qubits = num_levels
        self.backend = AerSimulator()
        self.statevector_backend = AerSimulator(method='statevector')
        self.noise_model = NoiseModel.from_backend(FakeMontrealV2()) if use_noise else None
        
    def create_universal_circuit(self) -> QuantumCircuit:
        """Create a universal quantum circuit for binomial distribution."""
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        theta = 2 * np.arccos(np.sqrt(self.bias))
        
        for i in range(self.num_qubits):
            qc.ry(theta, i)
        
        for layer in range(self.num_levels - 1):
            for i in range(self.num_levels - layer - 1):
                qc.cx(i, i + 1)
                qc.rz(np.pi/6, i + 1)
                qc.cx(i, i + 1)
        
        return qc
    
    def create_exponential_circuit(self, lambda_param: float = 1.0, precision_bits: int = 6) -> QuantumCircuit:
        """Create optimized quantum circuit for exponential distribution."""
        n_states = 2 ** precision_bits
        qc = QuantumCircuit(precision_bits, precision_bits)
        
        x_values = np.linspace(0, 5/lambda_param, n_states)
        probabilities = expon.pdf(x_values, scale=1/lambda_param)
        probabilities = probabilities / np.sum(probabilities)
        amplitudes = np.sqrt(probabilities)
        
        from qiskit.circuit.library import StatePreparation
        qc.append(StatePreparation(amplitudes), range(precision_bits))
        qc.measure_all()
        
        return qc
    
    def create_hadamard_walk_circuit(self, steps: int = None, position_bits: int = 5) -> QuantumCircuit:
        """Create optimized Hadamard quantum walk circuit."""
        if steps is None:
            steps = min(self.num_levels, 8)
        
        coin_qubit = 0
        position_qubits = list(range(1, position_bits + 1))
        qc = QuantumCircuit(position_bits + 1, position_bits + 1)
        
        center_pos = 2**(position_bits-1)
        for i, bit in enumerate(format(center_pos, f'0{position_bits}b')):
            if bit == '1':
                qc.x(i + 1)
        
        for step in range(steps):
            qc.h(coin_qubit)
            for i in range(position_bits - 1):
                qc.mcx([coin_qubit, position_bits - i - 1], position_bits - i - 2)
            qc.x(coin_qubit)
            for i in range(position_bits - 1, 0, -1):
                qc.mcx([coin_qubit, position_bits - i], position_bits - i - 1)
            qc.x(coin_qubit)
        
        qc.measure_all()
        return qc
    
    def simulate_distribution(self, circuit: QuantumCircuit, shots: int = 1000, apply_mitigation: bool = False) -> Dict[str, int]:
        """Simulate a quantum circuit with optional noise and error mitigation."""
        try:
            if apply_mitigation:
                cal_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
                cal_circuit.measure_all()
                cal_job = self.backend.run(cal_circuit, shots=shots, noise_model=self.noise_model)
                cal_results = cal_job.result()
                mitigator = LocalReadoutMitigator(backend=self.backend, noise_model=self.noise_model)
            else:
                mitigator = None
            
            job = self.backend.run(circuit, shots=shots, noise_model=self.noise_model)
            result = job.result()
            
            counts = result.get_counts()
            if apply_mitigation and mitigator:
                counts = mitigator.quasi_counts(counts)
            
            return counts
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}. Proceeding without mitigation.")
            job = self.backend.run(circuit, shots=shots, noise_model=self.noise_model)
            return job.result().get_counts()
    
    def simulate_binomial_distribution(self, shots: int = 1000, apply_mitigation: bool = False) -> Dict[str, int]:
        """Simulate binomial distribution."""
        circuit = self.create_universal_circuit()
        circuit.measure_all()
        return self.simulate_distribution(circuit, shots, apply_mitigation)
    
    def simulate_exponential_distribution(self, lambda_param: float = 1.0, shots: int = 1000, apply_mitigation: bool = False) -> Dict[str, int]:
        """Simulate exponential distribution."""
        circuit = self.create_exponential_circuit(lambda_param)
        return self.simulate_distribution(circuit, shots, apply_mitigation)
    
    def simulate_hadamard_walk(self, steps: int = None, shots: int = 1000, apply_mitigation: bool = False) -> Dict[str, int]:
        """Simulate Hadamard quantum walk with proper position decoding."""
        circuit = self.create_hadamard_walk_circuit(steps)
        return self.simulate_distribution(circuit, shots, apply_mitigation)
    
    def analyze_distribution(self, counts: Dict[str, int], distribution_type: str, **kwargs) -> Tuple[List[int], List[int], Dict]:
        """Analyze distribution results."""
        positions = []
        frequencies = []
        
        if distribution_type == 'hadamard_walk':
            position_bits = kwargs.get('position_bits', 5)
            for bitstring, count in counts.items():
                try:
                    parts = bitstring.split()
                    if len(parts) != 2:
                        warnings.warn(f"Invalid bitstring format '{bitstring}' for hadamard_walk, skipping")
                        continue
                    pos = int(parts[1], 2) - 2**(position_bits-1)
                    positions.append(pos)
                    frequencies.append(count)
                except ValueError as e:
                    warnings.warn(f"Skipping invalid bitstring '{bitstring}': {e}")
        else:
            for bitstring, count in counts.items():
                try:
                    bitstring_part = bitstring.split()[0] if ' ' in bitstring else bitstring
                    pos = int(bitstring_part, 2) if bitstring_part else 0
                    positions.append(pos)
                    frequencies.append(count)
                except ValueError as e:
                    warnings.warn(f"Skipping invalid bitstring '{bitstring}': {e}")
        
        total_shots = sum(frequencies)
        mean_pos = np.average(positions, weights=frequencies) if frequencies else 0
        variance = np.average([(pos - mean_pos)**2 for pos in positions], weights=frequencies) if frequencies else 0
        std_dev = np.sqrt(variance)
        
        return positions, frequencies, {'mean': mean_pos, 'variance': variance, 'std_dev': std_dev, 'total_shots': total_shots}
    
    def theoretical_distribution(self, distribution_type: str, **kwargs) -> Tuple[List[int], List[float]]:
        """Calculate theoretical distribution."""
        if distribution_type == 'binomial':
            positions = list(range(self.num_levels + 1))
            probabilities = [binom.pmf(k, self.num_levels, self.bias) for k in positions]
        elif distribution_type == 'exponential':
            lambda_param = kwargs.get('lambda_param', 1.0)
            precision_bits = kwargs.get('precision_bits', 6)
            n_states = 2 ** precision_bits
            x_values = np.linspace(0, 5/lambda_param, n_states)
            probabilities = expon.pdf(x_values, scale=1/lambda_param)
            probabilities = probabilities / np.sum(probabilities)
            positions = list(range(n_states))
        else:  # hadamard_walk
            steps = kwargs.get('steps', min(self.num_levels, 8))
            position_bits = kwargs.get('position_bits', 5)
            positions = list(range(-2**(position_bits-1), 2**(position_bits-1) + 1))
            probabilities = norm.pdf(positions, loc=0, scale=np.sqrt(steps))
            probabilities = probabilities / np.sum(probabilities)
        
        return positions, probabilities
    
    def plot_comprehensive_results(self, counts: Dict[str, int], distribution_type: str, shots: int, save_path: str = None, **kwargs):
        """Create comprehensive visualization with distance metrics."""
        positions, frequencies, stats = self.analyze_distribution(counts, distribution_type, **kwargs)
        theoretical_positions, theoretical_probs = self.theoretical_distribution(distribution_type, **kwargs)
        
        total_shots = stats['total_shots']
        metrics = compute_distance_metrics(counts, theoretical_probs, total_shots, max(len(theoretical_positions), max(positions, default=0) + 1))
        
        fig = plt.figure(figsize=(20, 12))
        
        ax1 = plt.subplot(2, 3, 1)
        ax1.bar(positions, frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{distribution_type.capitalize()} Distribution\nShots: {shots}')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        all_positions = theoretical_positions
        experimental_freqs = [0] * len(all_positions)
        for pos, freq in zip(positions, frequencies):
            if pos in all_positions:
                experimental_freqs[all_positions.index(pos)] = freq
        
        theoretical_freqs = [prob * total_shots for prob in theoretical_probs]
        ax2.bar([x - 0.2 for x in all_positions], experimental_freqs, width=0.4, alpha=0.7, color='skyblue', label='Quantum')
        ax2.bar([x + 0.2 for x in all_positions], theoretical_freqs, width=0.4, alpha=0.7, color='red', label='Theoretical')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Quantum vs Theoretical')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(2, 3, 3)
        experimental_probs = [freq / total_shots for freq in experimental_freqs] if total_shots > 0 else [0] * len(experimental_freqs)
        ax3.plot(all_positions, experimental_probs, 'o-', color='skyblue', label='Quantum')
        ax3.plot(theoretical_positions, theoretical_probs, 's-', color='red', label='Theoretical')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Probability')
        ax3.set_title('Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(2, 3, 4)
        errors = [abs(exp - theo) for exp, theo in zip(experimental_probs, theoretical_probs)]
        ax4.bar(all_positions, errors, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Error Analysis')
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        stats_text = f"""
        Statistics:
        Total Shots: {stats['total_shots']}
        Mean: {stats['mean']:.3f}
        Std Dev: {stats['std_dev']:.3f}
        KL Divergence: {metrics['kl_divergence']:.4f}
        JS Divergence: {metrics['js_divergence']:.4f}
        Total Variation: {metrics['total_variation']:.4f}
        """
        ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=12, verticalalignment='center', bbox=dict(facecolor='lightgray'))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_gaussian_convergence(self, levels_list: List[int] = None, shots: int = 5000, apply_mitigation: bool = False):
        """Test Gaussian convergence for binomial distribution across multiple levels."""
        if levels_list is None:
            levels_list = [4, 8, 12, 16]
        
        print(f"Testing Gaussian convergence with levels {levels_list}...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        original_levels = self.num_levels
        for i, levels in enumerate(levels_list):
            print(f"\nTesting {levels} levels...")
            self.num_levels = levels
            self.num_qubits = levels
            counts = self.simulate_binomial_distribution(shots, apply_mitigation)
            positions, frequencies, stats = self.analyze_distribution(counts, 'binomial')
            
            ax = axes[i]
            ax.bar(positions, frequencies, alpha=0.7, color='skyblue', label='Quantum Simulation')
            
            mean_theoretical = levels * self.bias
            std_theoretical = np.sqrt(levels * self.bias * (1 - self.bias))
            x = np.linspace(0, levels, 100)
            gaussian = norm.pdf(x, mean_theoretical, std_theoretical) * shots
            ax.plot(x, gaussian, 'r-', linewidth=2, label='Theoretical Gaussian')
            
            theoretical_probs = self.theoretical_distribution('binomial')[1]
            metrics = compute_distance_metrics(counts, theoretical_probs, stats['total_shots'], levels + 1)
            ax.text(0.05, 0.95, f'KL Div: {metrics["kl_divergence"]:.4f}', transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{levels} Levels\nμ={mean_theoretical:.1f}, σ={std_theoretical:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        self.num_levels = original_levels
        self.num_qubits = original_levels
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/gaussian_convergence_advanced_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gaussian convergence plot saved to: {save_path}")
    
    def test_noise_optimization(self, shots: int = 1000, noise_scales: List[float] = None, apply_mitigation: bool = False):
        """Test the effect of varying noise levels on binomial distribution."""
        if noise_scales is None:
            noise_scales = [0.0, 0.5, 1.0, 2.0]
        
        print(f"Testing noise optimization with scales {noise_scales}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        kl_divergences = []
        
        theoretical_positions, theoretical_probs = self.theoretical_distribution('binomial')
        original_noise_model = self.noise_model
        
        for scale in noise_scales:
            print(f"\nTesting noise scale: {scale}")
            if scale == 0.0:
                self.noise_model = None
            else:
                noise_model = NoiseModel.from_backend(FakeMontrealV2())
                scaled_noise = NoiseModel()
                for gate, errors in noise_model._local_quantum_errors.items():
                    scaled_errors = {}
                    for op, error in errors.items():
                        if isinstance(error, QuantumError) and error.to_dict()['type'] == 'depolarizing':
                            # Extract depolarizing probability and scale it
                            prob = error.to_dict()['instructions'][0].get('probabilities', [0])[1]
                            scaled_prob = min(prob * scale, 0.9999)  # Avoid prob=1.0
                            scaled_errors[op] = depolarizing_error(scaled_prob, error.num_qubits)
                        else:
                            warnings.warn(f"Unsupported error type for '{op}' on gate {gate}, using original error")
                            scaled_errors[op] = error
                    scaled_noise._local_quantum_errors[gate] = scaled_errors
                self.noise_model = scaled_noise
            
            counts = self.simulate_binomial_distribution(shots, apply_mitigation)
            positions, frequencies, stats = self.analyze_distribution(counts, 'binomial')
            
            ax1.bar([x + 0.2 * noise_scales.index(scale) for x in positions], frequencies, 
                    width=0.2, alpha=0.7, label=f'Noise Scale {scale}')
            
            metrics = compute_distance_metrics(counts, theoretical_probs, stats['total_shots'], self.num_levels + 1)
            kl_divergences.append(metrics['kl_divergence'])
        
        self.noise_model = original_noise_model
        ax1.plot(theoretical_positions, [prob * shots for prob in theoretical_probs], 
                 'r-', linewidth=2, label='Theoretical')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution vs Noise Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(noise_scales, kl_divergences, 'o-', color='orange')
        ax2.set_xlabel('Noise Scale')
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('KL Divergence vs Noise Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/noise_optimization_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise optimization plot saved to: {save_path}")
    
    def plot_uncertainty_analysis(self, uncertainty: Dict, distribution_type: str, save_path: str = None):
        """Plot uncertainty analysis for distance metrics."""
        metrics = ['kl_divergence', 'js_divergence', 'total_variation']
        means = [uncertainty[metric]['mean'] for metric in metrics]
        stds = [uncertainty[metric]['std'] for metric in metrics]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = range(len(metrics))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_ylabel('Value')
        ax.set_title(f'Uncertainty Analysis for {distribution_type.capitalize()} Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"output/uncertainty_analysis_{distribution_type}_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Uncertainty analysis plot saved to: {save_path}")
    
    def run_distribution_experiment(self, distribution_type: str = 'binomial', shots: int = 1000, 
                                   apply_mitigation: bool = False, num_runs: int = 5, 
                                   test_convergence: bool = False, test_noise: bool = False, **kwargs):
        """Run experiment for different distribution types with optional convergence and noise tests."""
        if shots < 1:
            raise ValueError("shots must be positive")
        print(f"Running {distribution_type} distribution experiment...")
        print(f"Shots: {shots}, Noise: {bool(self.noise_model)}, Mitigation: {apply_mitigation}")
        
        if distribution_type == 'binomial':
            counts = self.simulate_binomial_distribution(shots, apply_mitigation)
            theoretical_positions, theoretical_probs = self.theoretical_distribution(distribution_type)
            positions, frequencies, stats = self.analyze_distribution(counts, distribution_type)
        elif distribution_type == 'exponential':
            lambda_param = kwargs.get('lambda_param', 1.0)
            print(f"Lambda parameter: {lambda_param}")
            counts = self.simulate_exponential_distribution(lambda_param, shots, apply_mitigation)
            theoretical_positions, theoretical_probs = self.theoretical_distribution(distribution_type, lambda_param=lambda_param)
            positions, frequencies, stats = self.analyze_distribution(counts, distribution_type)
        elif distribution_type == 'hadamard_walk':
            steps = kwargs.get('steps', min(self.num_levels, 8))
            position_bits = kwargs.get('position_bits', 5)
            print(f"Walk steps: {steps}")
            counts = self.simulate_hadamard_walk(steps, shots, apply_mitigation)
            theoretical_positions, theoretical_probs = self.theoretical_distribution(distribution_type, steps=steps, position_bits=position_bits)
            positions, frequencies, stats = self.analyze_distribution(counts, distribution_type, position_bits=position_bits)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")
        
        uncertainty = compute_stochastic_uncertainty(
            self.simulate_binomial_distribution if distribution_type == 'binomial' else
            self.simulate_exponential_distribution if distribution_type == 'exponential' else
            self.simulate_hadamard_walk,
            num_runs, shots, theoretical_probs, len(theoretical_positions),
            apply_mitigation=apply_mitigation, **kwargs
        )
        
        print("\nResults:")
        print(f"Mean position: {stats['mean']:.3f}")
        print(f"Standard deviation: {stats['std_dev']:.3f}")
        print("\nDistance Metrics with Uncertainty:")
        for key, stats in uncertainty.items():
            print(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/{distribution_type}_distribution_{self.num_levels}levels_{timestamp}.pdf"
        self.plot_comprehensive_results(counts, distribution_type, shots, save_path, **kwargs)
        
        self.plot_uncertainty_analysis(uncertainty, distribution_type)
        
        if test_convergence and distribution_type == 'binomial':
            self.test_gaussian_convergence(shots=shots, apply_mitigation=apply_mitigation)
        if test_noise:
            self.test_noise_optimization(shots=shots, apply_mitigation=apply_mitigation)
        
        return {'counts': counts, 'positions': positions, 'frequencies': frequencies, 'stats': stats, 'uncertainty': uncertainty}

    def demonstrate_all_distributions(self, use_noise: bool, apply_mitigation: bool, shots: int = 1000):
        """Demonstrate all distribution types in a single figure."""
        print("=" * 70)
        print("DEMONSTRATING ALL DISTRIBUTION TYPES")
        print("=" * 70)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        distributions = [
            ('binomial', {}),
            ('exponential', {'lambda_param': 1.5}),
            ('hadamard_walk', {'steps': 6, 'position_bits': 5})
        ]
        
        for i, (dist_type, kwargs) in enumerate(distributions):
            print(f"\nSimulating {dist_type} distribution...")
            results = self.run_distribution_experiment(dist_type, shots, apply_mitigation, **kwargs)
            counts, positions, frequencies, stats = results['counts'], results['positions'], results['frequencies'], results['stats']
            
            ax = axes[i]
            ax.bar(positions, frequencies, alpha=0.7, color='skyblue', label='Quantum')
            theoretical_positions, theoretical_probs = self.theoretical_distribution(dist_type, **kwargs)
            theoretical_freqs = [prob * stats['total_shots'] for prob in theoretical_probs]
            ax.plot(theoretical_positions, theoretical_freqs, 'r-', linewidth=2, label='Theoretical')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{dist_type.capitalize()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/all_distributions_comparison_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"All distributions plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Quantum Galton Board Implementation')
    parser.add_argument('--levels', '-n', type=int, default=8)
    parser.add_argument('--bias', '-b', type=float, default=0.5)
    parser.add_argument('--shots', '-s', type=int, default=1000)
    parser.add_argument('--distribution', '-d', type=str, default='binomial', choices=['binomial', 'exponential', 'hadamard_walk'])
    parser.add_argument('--lambda-param', type=float, default=1.0)
    parser.add_argument('--walk-steps', type=int, default=None)
    parser.add_argument('--use-noise', action='store_true')
    parser.add_argument('--apply-mitigation', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--test-convergence', action='store_true', help='Test Gaussian convergence')
    parser.add_argument('--test-noise', action='store_true', help='Test noise optimization')
    
    args = parser.parse_args()
    
    if args.demo:
        galton_board = AdvancedQuantumGaltonBoard(num_levels=6, bias=0.5, use_noise=args.use_noise)
        galton_board.demonstrate_all_distributions(args.use_noise, args.apply_mitigation)
    else:
        print("Advanced Quantum Galton Board Implementation")
        print("=" * 70)
        galton_board = AdvancedQuantumGaltonBoard(num_levels=args.levels, bias=args.bias, use_noise=args.use_noise)
        if args.distribution == 'exponential':
            results = galton_board.run_distribution_experiment(
                args.distribution, args.shots, args.apply_mitigation, 
                test_convergence=args.test_convergence, test_noise=args.test_noise, lambda_param=args.lambda_param)
        elif args.distribution == 'hadamard_walk':
            results = galton_board.run_distribution_experiment(
                args.distribution, args.shots, args.apply_mitigation, 
                test_convergence=args.test_convergence, test_noise=args.test_noise, steps=args.walk_steps)
        else:
            results = galton_board.run_distribution_experiment(
                args.distribution, args.shots, args.apply_mitigation, 
                test_convergence=args.test_convergence, test_noise=args.test_noise)

if __name__ == "__main__":
    main()