import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram
import seaborn as sns
from typing import List, Tuple
import math
import argparse
from scipy.stats import norm

class QuantumGaltonBoard:
    """
    Quantum Galton Board implementation based on the paper:
    "Universal Statistical Simulator" by Mark Carney and Ben Varcoe
    
    This implements a quantum circuit that simulates the classical Galton Board
    using quantum superposition and measurement.
    """
    
    def __init__(self, num_levels: int = 8):
        """
        Initialize the Quantum Galton Board
        
        Args:
            num_levels: Number of levels in the Galton Board (default: 8)
        """
        self.num_levels = num_levels
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_galton_circuit(self, shots: int = 1000) -> QuantumCircuit:
        """
        Create the quantum circuit for the Galton Board
        
        The circuit implements a more accurate simulation of the Galton Board:
        1. Each level represents a row of pegs
        2. At each level, the ball can go left (0) or right (1)
        3. The final position is the sum of all right turns
        
        Args:
            shots: Number of shots for the simulation
            
        Returns:
            QuantumCircuit: The constructed quantum circuit
        """
        # We need num_levels qubits to represent each level
        qc = QuantumCircuit(self.num_levels, self.num_levels)
        
        # Apply Hadamard gates to create superposition at each level
        # Each Hadamard gate represents the 50/50 choice at each peg
        for i in range(self.num_levels):
            qc.h(i)
        
        # Measure all qubits
        # The number of 1s in the measurement result represents the final position
        qc.measure_all()
        
        return qc
    
    def simulate_galton_board(self, shots: int = 1000) -> dict:
        """
        Simulate the Quantum Galton Board
        
        Args:
            shots: Number of shots for the simulation
            
        Returns:
            dict: Measurement results with counts
        """
        qc = self.create_galton_circuit(shots)
        
        # Execute the circuit using legacy execute method
        job = self.backend.run(qc, shots=shots)
        result = job.result()
        
        return result.get_counts()
    
    def analyze_results(self, counts: dict) -> Tuple[List[int], List[int]]:
        """
        Analyze the measurement results to get the distribution
        
        Args:
            counts: Measurement counts from the simulation
            
        Returns:
            Tuple[List[int], List[int]]: Positions and their frequencies
        """
        positions = []
        frequencies = []
        
        for bitstring, count in counts.items():
            # Count the number of 1s in the bitstring (position)
            position = bitstring.count('1')
            positions.append(position)
            frequencies.append(count)
        
        return positions, frequencies
    
    def theoretical_binomial(self, n: int, p: float = 0.5) -> List[float]:
        """
        Calculate theoretical binomial distribution
        
        Args:
            n: Number of trials (levels)
            p: Probability of success (default: 0.5)
            
        Returns:
            List[float]: Theoretical probabilities for each position
        """
        probabilities = []
        for k in range(n + 1):
            prob = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
            probabilities.append(prob)
        return probabilities
    
    def plot_results(self, counts: dict, save_path: str = None):
        """
        Plot the results of the Quantum Galton Board simulation
        
        Args:
            counts: Measurement counts from the simulation
            save_path: Optional path to save the plot
        """
        positions, frequencies = self.analyze_results(counts)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Histogram of experimental results
        ax1.bar(positions, frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Position (Number of 1s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Quantum Galton Board - Experimental Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Comparison with theoretical binomial distribution and Gaussian
        theoretical_probs = self.theoretical_binomial(self.num_levels)
        total_shots = sum(frequencies)
        theoretical_freqs = [prob * total_shots for prob in theoretical_probs]
        
        # Create complete position arrays for comparison
        all_positions = list(range(self.num_levels + 1))
        
        # Create frequency arrays that match the complete position range
        experimental_freqs_complete = []
        for pos in all_positions:
            if pos in positions:
                idx = positions.index(pos)
                experimental_freqs_complete.append(frequencies[idx])
            else:
                experimental_freqs_complete.append(0)
        
        ax2.bar([x - 0.2 for x in all_positions], experimental_freqs_complete, width=0.4, 
                alpha=0.7, color='skyblue', label='Quantum Simulation', edgecolor='black')
        ax2.bar([x + 0.2 for x in all_positions], theoretical_freqs, width=0.4,
                alpha=0.7, color='red', label='Theoretical Binomial', edgecolor='black')
        
        # Add Gaussian approximation
        mean_theoretical = self.num_levels * 0.5
        std_theoretical = np.sqrt(self.num_levels * 0.25)
        x_gauss = np.linspace(0, self.num_levels, 100)
        gaussian = norm.pdf(x_gauss, mean_theoretical, std_theoretical)
        
        # Scale Gaussian to match the frequency scale
        gaussian_scaled = gaussian * (max(theoretical_freqs) / max(gaussian))
        ax2.plot(x_gauss, gaussian_scaled, 'g-', linewidth=2, label='Gaussian Approximation', alpha=0.7)
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Comparison: Quantum vs Theoretical vs Gaussian')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()  # 表示を無効化
    
    def run_experiment(self, shots: int = 1000, plot: bool = True, save_path: str = None):
        """
        Run a complete Quantum Galton Board experiment
        
        Args:
            shots: Number of shots for the simulation
            plot: Whether to plot the results
            save_path: Optional path to save the plot
            
        Returns:
            dict: Measurement results
        """
        print(f"Running Quantum Galton Board simulation with {shots} shots...")
        print(f"Number of levels: {self.num_levels}")
        
        # Run simulation
        counts = self.simulate_galton_board(shots)
        
        # Analyze results
        positions, frequencies = self.analyze_results(counts)
        
        print(f"\nResults:")
        print(f"Total measurements: {sum(frequencies)}")
        print(f"Position distribution: {dict(zip(positions, frequencies))}")
        
        # Calculate statistics
        mean_pos = np.average(positions, weights=frequencies)
        variance_pos = np.average([(pos - mean_pos)**2 for pos in positions], weights=frequencies)
        std_pos = np.sqrt(variance_pos)
        
        theoretical_mean = self.num_levels * 0.5
        theoretical_std = np.sqrt(self.num_levels * 0.25)
        
        print(f"Mean position: {mean_pos:.2f}")
        print(f"Theoretical mean: {theoretical_mean:.2f}")
        print(f"Standard deviation: {std_pos:.2f}")
        print(f"Theoretical std: {theoretical_std:.2f}")
        print(f"Mean error: {abs(mean_pos - theoretical_mean):.3f}")
        print(f"Std error: {abs(std_pos - theoretical_std):.3f}")
        
        if plot:
            # Generate default save path if not provided
            if save_path is None:
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"output/quantum_galton_board_{self.num_levels}levels_{shots}shots_{timestamp}.pdf"
            
            self.plot_results(counts, save_path)
        
        return counts
    
    def test_gaussian_convergence(self, levels_list: list = None, shots: int = 5000):
        """
        Test Gaussian convergence with multiple levels
        
        Args:
            levels_list: List of levels to test
            shots: Number of shots for each simulation
        """
        if levels_list is None:
            levels_list = [4, 8, 12, 16]
        
        print("Testing Gaussian convergence with multiple levels...")
        print(f"Levels to test: {levels_list}")
        print(f"Shots per simulation: {shots}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, levels in enumerate(levels_list):
            print(f"\nTesting {levels} levels...")
            
            try:
                # Create new Galton Board for this level
                galton_board = QuantumGaltonBoard(num_levels=levels)
                counts = galton_board.simulate_galton_board(shots)
                positions, frequencies = galton_board.analyze_results(counts)
                
                # Calculate theoretical Gaussian parameters
                mean_theoretical = levels * 0.5
                std_theoretical = np.sqrt(levels * 0.25)
                
                # Create theoretical Gaussian
                x = np.linspace(0, levels, 100)
                gaussian = norm.pdf(x, mean_theoretical, std_theoretical) * shots
                
                # Plot results
                ax = axes[i]
                
                # Experimental results
                ax.bar(positions, frequencies, alpha=0.7, color='skyblue', 
                       label='Quantum Simulation', edgecolor='black')
                
                # Theoretical Gaussian
                ax.plot(x, gaussian, 'r-', linewidth=2, label='Theoretical Gaussian')
                
                ax.set_xlabel('Position')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{levels} Levels\nμ={mean_theoretical:.1f}, σ={std_theoretical:.2f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Calculate and print statistics
                mean_exp = np.average(positions, weights=frequencies)
                variance_exp = np.average([(pos - mean_exp)**2 for pos in positions], weights=frequencies)
                std_exp = np.sqrt(variance_exp)
                
                print(f"  Experimental: Mean={mean_exp:.2f}, Std={std_exp:.2f}")
                print(f"  Theoretical:  Mean={mean_theoretical:.2f}, Std={std_theoretical:.2f}")
                print(f"  Error:        Mean={abs(mean_exp-mean_theoretical):.3f}, Std={abs(std_exp-std_theoretical):.3f}")
                
            except Exception as e:
                print(f"  Error with {levels} levels: {e}")
                ax = axes[i]
                ax.text(0.5, 0.5, f'Error: {levels} levels\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{levels} Levels - Failed')
        
        plt.tight_layout()
        
        # Save the plot
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/gaussian_convergence_test_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGaussian convergence test saved to: {save_path}")
    
    def test_theoretical_convergence(self, levels_list: list = None):
        """
        Test theoretical convergence to Gaussian without quantum simulation
        
        Args:
            levels_list: List of levels to test
        """
        if levels_list is None:
            levels_list = [10, 20, 50, 100]
        
        print("\nTesting theoretical convergence to Gaussian...")
        print(f"Levels to analyze: {levels_list}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, levels in enumerate(levels_list):
            print(f"\nAnalyzing {levels} levels theoretically...")
            
            # Calculate theoretical parameters
            mean_theoretical = levels * 0.5
            std_theoretical = np.sqrt(levels * 0.25)
            
            # Create theoretical binomial distribution
            from scipy.stats import binom
            x_binom = np.arange(levels + 1)
            binom_probs = binom.pmf(x_binom, levels, 0.5)
            
            # Create theoretical Gaussian
            x_gauss = np.linspace(0, levels, 200)
            gaussian = norm.pdf(x_gauss, mean_theoretical, std_theoretical)
            
            # Plot results
            ax = axes[i]
            
            # Binomial distribution
            ax.plot(x_binom, binom_probs, 'bo-', markersize=4, label='Binomial', alpha=0.7)
            
            # Gaussian approximation
            ax.plot(x_gauss, gaussian, 'r-', linewidth=2, label='Gaussian')
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Probability')
            ax.set_title(f'{levels} Levels\nμ={mean_theoretical:.1f}, σ={std_theoretical:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate convergence metric
            # Compare binomial and Gaussian at integer points
            gaussian_at_integers = norm.pdf(x_binom, mean_theoretical, std_theoretical)
            mse = np.mean((binom_probs - gaussian_at_integers)**2)
            print(f"  MSE between Binomial and Gaussian: {mse:.6f}")
            print(f"  Standard deviation: {std_theoretical:.2f}")
        
        plt.tight_layout()
        
        # Save the plot
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/theoretical_convergence_{timestamp}.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTheoretical convergence test saved to: {save_path}")

def main():
    """
    Main function to demonstrate the Quantum Galton Board
    """
    parser = argparse.ArgumentParser(description='Quantum Galton Board Implementation')
    parser.add_argument('--levels', '-n', type=int, default=8, 
                       help='Number of levels in the Galton Board (default: 8)')
    parser.add_argument('--shots', '-s', type=int, default=1000,
                       help='Number of shots for simulation (default: 1000)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting (only save to file)')
    parser.add_argument('--convergence', action='store_true',
                       help='Test Gaussian convergence with multiple levels')
    parser.add_argument('--theoretical', action='store_true',
                       help='Test theoretical convergence to Gaussian')
    parser.add_argument('--convergence-levels', nargs='+', type=int,
                       help='Custom levels for convergence test')
    parser.add_argument('--theoretical-levels', nargs='+', type=int,
                       help='Custom levels for theoretical test')
    
    args = parser.parse_args()
    
    print("Quantum Galton Board Implementation")
    print("Based on: Universal Statistical Simulator by Mark Carney and Ben Varcoe")
    print("=" * 60)
    
    if args.convergence:
        print("Running Gaussian convergence test...")
        galton_board = QuantumGaltonBoard(num_levels=8)  # Default level for initialization
        galton_board.test_gaussian_convergence(
            levels_list=args.convergence_levels,
            shots=args.shots
        )
    elif args.theoretical:
        print("Running theoretical convergence test...")
        galton_board = QuantumGaltonBoard(num_levels=8)  # Default level for initialization
        galton_board.test_theoretical_convergence(
            levels_list=args.theoretical_levels
        )
    else:
        print(f"Parameters: Levels={args.levels}, Shots={args.shots}")
        print("=" * 60)
        
        # Create and run the Quantum Galton Board
        galton_board = QuantumGaltonBoard(num_levels=args.levels)
        
        # Run experiment
        galton_board.run_experiment(shots=args.shots, plot=not args.no_plot)
        print(f"Graph saved to output folder")

if __name__ == "__main__":
    main() 
