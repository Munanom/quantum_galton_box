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
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
import seaborn as sns
from typing import List, Tuple
import math
import argparse
from scipy.stats import norm, binom
from utils import compute_distance_metrics, compute_stochastic_uncertainty

class QuantumGaltonBoard:
    """Quantum Galton Board implementation with noise model and error mitigation."""
    
    def __init__(self, num_levels: int = 8, use_noise: bool = False):
        self.num_levels = num_levels
        self.backend = Aer.get_backend('qasm_simulator')
        self.noise_model = NoiseModel.from_backend(FakeMontrealV2()) if use_noise else None
        
    def create_galton_circuit(self) -> QuantumCircuit:
        """Create the quantum circuit for the Galton Board."""
        qc = QuantumCircuit(self.num_levels, self.num_levels)
        for i in range(self.num_levels):
            qc.h(i)
        qc.measure_all()
        return qc
    
    def simulate_galton_board(self, shots: int = 1000, apply_mitigation: bool = False) -> dict:
        """Simulate the Quantum Galton Board with optional noise and error mitigation."""
        qc = self.create_galton_circuit()
        
        if apply_mitigation and CompleteMeasFitter is not None:
            # Create a calibration circuit for measurement error mitigation
            cal_circuit = QuantumCircuit(self.num_levels, self.num_levels)
            cal_circuit.measure_all()
            cal_job = self.backend.run(cal_circuit, shots=shots, noise_model=self.noise_model)
            cal_results = cal_job.result()
            meas_fitter = CompleteMeasFitter(cal_results, [format(i, f'0{self.num_levels}b') for i in range(2**self.num_levels)])
        else:
            meas_fitter = None
            if apply_mitigation:
                print("Warning: Measurement mitigation unavailable, proceeding without.")
        
        job = self.backend.run(qc, shots=shots, noise_model=self.noise_model)
        result = job.result()
        
        counts = result.get_counts()
        if apply_mitigation and meas_fitter:
            counts = meas_fitter.filter.apply(result).get_counts()
        
        return counts
    
    def analyze_results(self, counts: dict) -> Tuple[List[int], List[int]]:
        """Analyze the measurement results to get the distribution."""
        positions = []
        frequencies = []
        
        for bitstring, count in counts.items():
            position = bitstring.count('1')
            positions.append(position)
            frequencies.append(count)
        
        return positions, frequencies
    
    def theoretical_binomial(self, n: int, p: float = 0.5) -> List[float]:
        """Calculate theoretical binomial distribution."""
        return [binom.pmf(k, n, p) for k in range(n + 1)]
    
    def plot_results(self, counts: dict, save_path: str = None):
        """Plot the results with comparison to theoretical binomial and Gaussian."""
        positions, frequencies = self.analyze_results(counts)
        total_shots = sum(frequencies)
        
        # Compute distance metrics
        theoretical_probs = np.array(self.theoretical_binomial(self.num_levels))
        metrics = compute_distance_metrics(counts, theoretical_probs, total_shots, self.num_levels + 1)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Histogram
        ax1.bar(positions, frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Position (Number of 1s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Quantum Galton Board - {self.num_levels} Levels')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Comparison
        all_positions = list(range(self.num_levels + 1))
        experimental_freqs = [0] * (self.num_levels + 1)
        for pos, freq in zip(positions, frequencies):
            experimental_freqs[pos] = freq
        
        theoretical_freqs = [prob * total_shots for prob in theoretical_probs]
        ax2.bar([x - 0.2 for x in all_positions], experimental_freqs, width=0.4, 
                alpha=0.7, color='skyblue', label='Quantum Simulation', edgecolor='black')
        ax2.bar([x + 0.2 for x in all_positions], theoretical_freqs, width=0.4,
                alpha=0.7, color='red', label='Theoretical Binomial', edgecolor='black')
        
        # Gaussian approximation
        mean_theoretical = self.num_levels * 0.5
        std_theoretical = np.sqrt(self.num_levels * 0.25)
        x_gauss = np.linspace(0, self.num_levels, 100)
        gaussian = norm.pdf(x_gauss, mean_theoretical, std_theoretical) * total_shots
        ax2.plot(x_gauss, gaussian, 'g-', linewidth=2, label='Gaussian Approximation')
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Comparison: Quantum vs Theoretical vs Gaussian')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add metrics to plot
        metrics_text = f"KL Divergence: {metrics['kl_divergence']:.4f}\nJS Divergence: {metrics['js_divergence']:.4f}\nTotal Variation: {metrics['total_variation']:.4f}"
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_experiment(self, shots: int = 1000, plot: bool = True, save_path: str = None, apply_mitigation: bool = False, num_runs: int = 5):
        """Run a complete Quantum Galton Board experiment with stochastic uncertainty."""
        print(f"Running Quantum Galton Board with {shots} shots, {self.num_levels} levels, noise={bool(self.noise_model)}...")
        
        counts = self.simulate_galton_board(shots, apply_mitigation)
        positions, frequencies = self.analyze_results(counts)
        
        # Calculate statistics
        mean_pos = np.average(positions, weights=frequencies)
        variance_pos = np.average([(pos - mean_pos)**2 for pos in positions], weights=frequencies)
        std_pos = np.sqrt(variance_pos)
        theoretical_mean = self.num_levels * 0.5
        theoretical_std = np.sqrt(self.num_levels * 0.25)
        
        print(f"Total measurements: {sum(frequencies)}")
        print(f"Mean position: {mean_pos:.2f} (Theoretical: {theoretical_mean:.2f})")
        print(f"Standard deviation: {std_pos:.2f} (Theoretical: {theoretical_std:.2f})")
        
        # Compute stochastic uncertainty
        theoretical_probs = np.array(self.theoretical_binomial(self.num_levels))
        uncertainty = compute_stochastic_uncertainty(
            self.simulate_galton_board, num_runs, shots, theoretical_probs, self.num_levels + 1, apply_mitigation=apply_mitigation
        )
        print("\nDistance Metrics with Uncertainty:")
        for key, stats in uncertainty.items():
            print(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if plot:
            if save_path is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"output/quantum_galton_board_{self.num_levels}levels_{shots}shots_{timestamp}.pdf"
            self.plot_results(counts, save_path)
        
        return counts
    
    def test_gaussian_convergence(self, levels_list: list = None, shots: int = 5000, apply_mitigation: bool = False):
        """Test Gaussian convergence with multiple levels."""
        if levels_list is None:
            levels_list = [4, 8, 12, 16]
        
        print(f"Testing Gaussian convergence with levels {levels_list}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, levels in enumerate(levels_list):
            print(f"\nTesting {levels} levels...")
            galton_board = QuantumGaltonBoard(num_levels=levels, use_noise=bool(self.noise_model))
            counts = galton_board.simulate_galton_board(shots, apply_mitigation)
            positions, frequencies = galton_board.analyze_results(counts)
            
            # Plot results
            ax = axes[i]
            ax.bar(positions, frequencies, alpha=0.7, color='skyblue', label='Quantum Simulation')
            
            # Theoretical Gaussian
            mean_theoretical = levels * 0.5
            std_theoretical = np.sqrt(levels * 0.25)
            x = np.linspace(0, levels, 100)
            gaussian = norm.pdf(x, mean_theoretical, std_theoretical) * shots
            ax.plot(x, gaussian, 'r-', linewidth=2, label='Theoretical Gaussian')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{levels} Levels\nμ={mean_theoretical:.1f}, σ={std_theoretical:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/gaussian_convergence_test_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gaussian convergence test saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Quantum Galton Board Implementation')
    parser.add_argument('--levels', '-n', type=int, default=8)
    parser.add_argument('--shots', '-s', type=int, default=1000)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--convergence', action='store_true')
    parser.add_argument('--convergence-levels', nargs='+', type=int)
    parser.add_argument('--use-noise', action='store_true')
    parser.add_argument('--apply-mitigation', action='store_true')
    
    args = parser.parse_args()
    
    print("Quantum Galton Board Implementation")
    print("=" * 60)
    
    if args.convergence:
        galton_board = QuantumGaltonBoard(num_levels=8, use_noise=args.use_noise)
        galton_board.test_gaussian_convergence(args.convergence_levels, args.shots, args.apply_mitigation)
    else:
        galton_board = QuantumGaltonBoard(num_levels=args.levels, use_noise=args.use_noise)
        galton_board.run_experiment(args.shots, plot=not args.no_plot, apply_mitigation=args.apply_mitigation)
        print(f"Graph saved to output folder")

if __name__ == "__main__":
    main()