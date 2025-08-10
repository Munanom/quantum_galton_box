"""
Implements an advanced Quantum Galton Board based on the 'Universal Statistical Simulator' by Carney and Varcoe. The AdvancedQuantumGaltonBoard class:
- Simulates statistical distributions (binomial, exponential, Hadamard quantum walk) using quantum circuits.
- Creates a universal quantum circuit with rotation and entanglement operations to mimic a Galton Board.
- Simulates distributions with measurement and statevector analysis.
- Provides comprehensive visualization including experimental histograms, theoretical comparisons, error analysis, and statistics.
- Supports command-line arguments for customizing levels, bias, shots, and distribution types.
- Includes a demonstration mode to showcase all supported distributions.
The module extends the basic Galton Board simulation with flexible distribution types and detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeMontrealV2  # Using FakeMontrealV2 (27 qubits)
try:
    from qiskit.utils.mitigation import CompleteMeasFitter  # Updated import
except ImportError:
    print("Warning: CompleteMeasFitter not found. Disabling measurement mitigation.")
    CompleteMeasFitter = None  # Fallback for missing mitigation
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_histogram
import seaborn as sns
from typing import List, Tuple, Dict
import math
from scipy.stats import binom, expon, norm
import argparse
from utils import compute_distance_metrics, compute_stochastic_uncertainty

class AdvancedQuantumGaltonBoard:
    """Advanced Quantum Galton Board with noise model and multiple distributions."""
    
    def __init__(self, num_levels: int = 8, bias: float = 0.5, use_noise: bool = False):
        self.num_levels = num_levels
        self.bias = bias
        self.num_qubits = num_levels
        self.backend = Aer.get_backend('qasm_simulator')
        self.statevector_backend = Aer.get_backend('statevector_simulator')
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
        
        # Optimize: Use efficient state preparation
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
        
        # Initialize position at center
        center_pos = 2**(position_bits-1)
        for i, bit in enumerate(format(center_pos, f'0{position_bits}b')):
            if bit == '1':
                qc.x(i + 1)
        
        # Optimize: Reduce gate count in shift operation
        for step in range(steps):
            qc.h(coin_qubit)
            # Simplified shift using multi-controlled gates
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
        if apply_mitigation and CompleteMeasFitter is not None:
            cal_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
            cal_circuit.measure_all()
            cal_job = self.backend.run(cal_circuit, shots=shots, noise_model=self.noise_model)
            cal_results = cal_job.result()
            meas_fitter = CompleteMeasFitter(cal_results, [format(i, f'0{circuit.num_qubits}b') for i in range(2**circuit.num_qubits)])
        else:
            meas_fitter = None
            if apply_mitigation:
                print("Warning: Measurement mitigation unavailable, proceeding without.")
        
        job = self.backend.run(circuit, shots=shots, noise_model=self.noise_model)
        result = job.result()
        
        counts = result.get_counts()
        if apply_mitigation and meas_fitter:
            counts = meas_fitter.filter.apply(result).get_counts()
        
        return counts
    
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
                # Decode position as signed integer
                pos = int(bitstring[1:], 2) - 2**(position_bits-1)
                positions.append(pos)
                frequencies.append(count)
        else:
            for bitstring, count in counts.items():
                pos = int(bitstring, 2) if bitstring else 0
                positions.append(pos)
                frequencies.append(count)
        
        total_shots = sum(frequencies)
        mean_pos = np.average(positions, weights=frequencies)
        variance = np.average([(pos - mean_pos)**2 for pos in positions], weights=frequencies)
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
            # Simplified theoretical Hadamard walk distribution
            probabilities = norm.pdf(positions, loc=0, scale=np.sqrt(steps))  # Approximation
            probabilities = probabilities / np.sum(probabilities)
        
        return positions, probabilities
    
    def plot_comprehensive_results(self, counts: Dict[str, int], distribution_type: str, shots: int, save_path: str = None, **kwargs):
        """Create comprehensive visualization with distance metrics."""
        positions, frequencies, stats = self.analyze_distribution(counts, distribution_type, **kwargs)
        theoretical_positions, theoretical_probs = self.theoretical_distribution(distribution_type, **kwargs)
        
        total_shots = stats['total_shots']
        metrics = compute_distance_metrics(counts, theoretical_probs, total_shots, max(len(theoretical_positions), max(positions) + 1))
        
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Experimental results
        ax1 = plt.subplot(2, 3, 1)
        ax1.bar(positions, frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{distribution_type.capitalize()} Distribution\nShots: {shots}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Comparison
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
        
        # Plot 3: Probability comparison
        ax3 = plt.subplot(2, 3, 3)
        experimental_probs = [freq / total_shots for freq in experimental_freqs]
        ax3.plot(all_positions, experimental_probs, 'o-', color='skyblue', label='Quantum')
        ax3.plot(theoretical_positions, theoretical_probs, 's-', color='red', label='Theoretical')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Probability')
        ax3.set_title('Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error analysis
        ax4 = plt.subplot(2, 3, 4)
        errors = [abs(exp - theo) for exp, theo in zip(experimental_probs, theoretical_probs)]
        ax4.bar(all_positions, errors, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Error Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Statistics summary
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
    
    def run_distribution_experiment(self, distribution_type: str = 'binomial', shots: int = 1000, apply_mitigation: bool = False, num_runs: int = 5, **kwargs):
        """Run experiment for different distribution types with uncertainty analysis."""
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
        
        # Compute stochastic uncertainty
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
        
        # Plot results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/{distribution_type}_distribution_{self.num_levels}levels_{timestamp}.pdf"
        self.plot_comprehensive_results(counts, distribution_type, shots, save_path, **kwargs)
        
        return {'counts': counts, 'positions': positions, 'frequencies': frequencies, 'stats': stats, 'uncertainty': uncertainty}

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
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_all_distributions(args.use_noise, args.apply_mitigation)
    else:
        print("Advanced Quantum Galton Board Implementation")
        print("=" * 70)
        galton_board = AdvancedQuantumGaltonBoard(num_levels=args.levels, bias=args.bias, use_noise=args.use_noise)
        if args.distribution == 'exponential':
            results = galton_board.run_distribution_experiment(args.distribution, args.shots, args.apply_mitigation, lambda_param=args.lambda_param)
        elif args.distribution == 'hadamard_walk':
            results = galton_board.run_distribution_experiment(args.distribution, args.shots, args.apply_mitigation, steps=args.walk_steps)
        else:
            results = galton_board.run_distribution_experiment(args.distribution, args.shots, args.apply_mitigation)

def demonstrate_all_distributions(use_noise: bool, apply_mitigation: bool):
    """Demonstrate all distribution types."""
    print("=" * 70)
    print("DEMONSTRATING ALL DISTRIBUTION TYPES")
    print("=" * 70)
    
    galton_board = AdvancedQuantumGaltonBoard(num_levels=6, bias=0.5, use_noise=use_noise)
    
    print("\n1. BINOMIAL DISTRIBUTION:")
    galton_board.run_distribution_experiment('binomial', shots=1000, apply_mitigation=apply_mitigation)
    
    print("\n2. EXPONENTIAL DISTRIBUTION:")
    galton_board.run_distribution_experiment('exponential', shots=1000, apply_mitigation=apply_mitigation, lambda_param=1.5)
    
    print("\n3. HADAMARD QUANTUM WALK:")
    galton_board.run_distribution_experiment('hadamard_walk', shots=1000, apply_mitigation=apply_mitigation, steps=6)
    
    print("\n" + "=" * 70)
    print("All distributions completed! Check output folder for plots.")

if __name__ == "__main__":
    main()