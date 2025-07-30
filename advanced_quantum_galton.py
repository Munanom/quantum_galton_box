import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Operator, Statevector
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import seaborn as sns
from typing import List, Tuple, Dict
import math
from scipy.stats import binom
import argparse

class AdvancedQuantumGaltonBoard:
    """
    Advanced Quantum Galton Board implementation based on the paper:
    "Universal Statistical Simulator" by Mark Carney and Ben Varcoe
    
    This implementation more closely follows the paper's approach of using
    quantum circuits to simulate statistical distributions.
    """
    
    def __init__(self, num_levels: int = 8, bias: float = 0.5):
        """
        Initialize the Advanced Quantum Galton Board
        
        Args:
            num_levels: Number of levels in the Galton Board
            bias: Bias parameter for the distribution (0.5 = fair, <0.5 = left bias, >0.5 = right bias)
        """
        self.num_levels = num_levels
        self.bias = bias
        self.num_qubits = num_levels
        self.backend = Aer.get_backend('qasm_simulator')
        self.statevector_backend = Aer.get_backend('statevector_simulator')
        
    def create_universal_circuit(self) -> QuantumCircuit:
        """
        Create a universal quantum circuit that can simulate various statistical distributions
        
        This circuit implements the core idea from the paper where quantum
        superposition and entanglement can be used to generate statistical
        distributions that match classical probability theory.
        
        Returns:
            QuantumCircuit: The universal statistical simulator circuit
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Initialize all qubits in |0⟩ state
        # Apply rotation gates to create the desired bias
        theta = 2 * np.arccos(np.sqrt(self.bias))
        
        for i in range(self.num_qubits):
            qc.ry(theta, i)
        
        # Apply entanglement layers to create correlations
        # This simulates the cascading effect of the Galton Board
        for layer in range(self.num_levels - 1):
            for i in range(self.num_levels - layer - 1):
                # Controlled rotation to create entanglement
                qc.cx(i, i + 1)
                qc.rz(np.pi/6, i + 1)
                qc.cx(i, i + 1)
        
        return qc
    
    def create_measurement_circuit(self, base_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Add measurement operations to the base circuit
        
        Args:
            base_circuit: The base quantum circuit
            
        Returns:
            QuantumCircuit: Circuit with measurement operations
        """
        qc = base_circuit.copy()
        qc.measure_all()
        return qc
    
    def simulate_distribution(self, shots: int = 1000) -> Dict[str, int]:
        """
        Simulate the statistical distribution using the quantum circuit
        
        Args:
            shots: Number of shots for the simulation
            
        Returns:
            Dict[str, int]: Measurement results with counts
        """
        base_circuit = self.create_universal_circuit()
        measurement_circuit = self.create_measurement_circuit(base_circuit)
        
        # Execute the circuit using legacy execute method
        job = self.backend.run(measurement_circuit, shots=shots)
        result = job.result()
        
        return result.get_counts()
    
    def get_statevector(self) -> Statevector:
        """
        Get the statevector of the quantum system before measurement
        
        Returns:
            Statevector: The quantum state vector
        """
        # Check if circuit is too large for statevector simulation
        if self.num_qubits > 20:
            raise ValueError(f"Statevector simulation requires too much memory for {self.num_qubits} qubits. "
                           f"Maximum recommended: 20 qubits (requires ~16GB RAM)")
        
        base_circuit = self.create_universal_circuit()
        job = execute(base_circuit, self.statevector_backend)
        result = job.result()
        return result.get_statevector()
    
    def analyze_distribution(self, counts: Dict[str, int]) -> Tuple[List[int], List[int], Dict]:
        """
        Analyze the measurement results and calculate statistics
        
        Args:
            counts: Measurement counts from the simulation
            
        Returns:
            Tuple containing positions, frequencies, and statistics
        """
        positions = []
        frequencies = []
        
        for bitstring, count in counts.items():
            # Count the number of 1s in the bitstring
            position = bitstring.count('1')
            positions.append(position)
            frequencies.append(count)
        
        # Calculate statistics
        total_shots = sum(frequencies)
        mean_pos = np.average(positions, weights=frequencies)
        variance = np.average([(pos - mean_pos)**2 for pos in positions], weights=frequencies)
        std_dev = np.sqrt(variance)
        
        stats = {
            'mean': mean_pos,
            'variance': variance,
            'std_dev': std_dev,
            'total_shots': total_shots
        }
        
        return positions, frequencies, stats
    
    def theoretical_distribution(self) -> Tuple[List[int], List[float]]:
        """
        Calculate the theoretical binomial distribution
        
        Returns:
            Tuple[List[int], List[float]]: Positions and their theoretical probabilities
        """
        positions = list(range(self.num_levels + 1))
        probabilities = [binom.pmf(k, self.num_levels, self.bias) for k in positions]
        return positions, probabilities
    
    def plot_comprehensive_results(self, counts: Dict[str, int], save_path: str = None):
        """
        Create comprehensive visualization of the results
        
        Args:
            counts: Measurement counts from the simulation
            save_path: Optional path to save the plot
        """
        positions, frequencies, stats = self.analyze_distribution(counts)
        theoretical_positions, theoretical_probs = self.theoretical_distribution()
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Experimental results histogram
        ax1 = plt.subplot(2, 3, 1)
        ax1.bar(positions, frequencies, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Position (Number of 1s)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Quantum Simulation Results\n(Shots: {stats["total_shots"]})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Theoretical vs Experimental comparison
        ax2 = plt.subplot(2, 3, 2)
        theoretical_freqs = [prob * stats['total_shots'] for prob in theoretical_probs]
        
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
                alpha=0.7, color='skyblue', label='Quantum', edgecolor='black')
        ax2.bar([x + 0.2 for x in all_positions], theoretical_freqs, width=0.4,
                alpha=0.7, color='red', label='Theoretical', edgecolor='black')
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Quantum vs Theoretical Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Probability comparison
        ax3 = plt.subplot(2, 3, 3)
        experimental_probs = [freq / stats['total_shots'] for freq in experimental_freqs_complete]
        
        ax3.plot(all_positions, experimental_probs, 'o-', color='skyblue', label='Quantum', linewidth=2)
        ax3.plot(theoretical_positions, theoretical_probs, 's-', color='red', label='Theoretical', linewidth=2)
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Probability')
        ax3.set_title('Probability Distribution Comparison')
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
        Statistics Summary:
        
        Total Shots: {stats['total_shots']}
        Mean Position: {stats['mean']:.3f}
        Theoretical Mean: {self.num_levels * self.bias:.3f}
        Variance: {stats['variance']:.3f}
        Standard Deviation: {stats['std_dev']:.3f}
        Theoretical Std Dev: {np.sqrt(self.num_levels * self.bias * (1 - self.bias)):.3f}
        
        Bias Parameter: {self.bias}
        Number of Levels: {self.num_levels}
        """
        
        ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # Plot 6: Circuit visualization (simplified)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        circuit_info = f"""
        Quantum Circuit Info:
        
        Number of Qubits: {self.num_qubits}
        Circuit Depth: {self.num_levels}
        Bias Parameter: {self.bias}
        
        Circuit Operations:
        - RY gates for bias
        - CX gates for entanglement
        - RZ gates for phase
        - Measurement
        """
        
        ax6.text(0.1, 0.5, circuit_info, transform=ax6.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()  # 表示を無効化
    
    def run_comprehensive_experiment(self, shots: int = 1000, plot: bool = True, save_path: str = None):
        """
        Run a comprehensive Quantum Galton Board experiment
        
        Args:
            shots: Number of shots for the simulation
            plot: Whether to plot the results
            save_path: Optional path to save the plot
            
        Returns:
            dict: Complete experiment results
        """
        print(f"Advanced Quantum Galton Board Experiment")
        print(f"Number of levels: {self.num_levels}")
        print(f"Bias parameter: {self.bias}")
        print(f"Shots: {shots}")
        print("=" * 60)
        
        # Run simulation
        counts = self.simulate_distribution(shots)
        
        # Analyze results
        positions, frequencies, stats = self.analyze_distribution(counts)
        
        print(f"\nResults:")
        print(f"Total measurements: {stats['total_shots']}")
        print(f"Mean position: {stats['mean']:.3f}")
        print(f"Theoretical mean: {self.num_levels * self.bias:.3f}")
        print(f"Standard deviation: {stats['std_dev']:.3f}")
        print(f"Theoretical std dev: {np.sqrt(self.num_levels * self.bias * (1 - self.bias)):.3f}")
        
        if plot:
            # Generate default save path if not provided
            if save_path is None:
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"output/advanced_quantum_galton_{self.num_levels}levels_bias{self.bias}_{shots}shots_{timestamp}.pdf"
            
            self.plot_comprehensive_results(counts, save_path)
        
        return {
            'counts': counts,
            'positions': positions,
            'frequencies': frequencies,
            'stats': stats
        }

def demonstrate_universal_simulator():
    """
    Demonstrate the universal statistical simulator capabilities
    """
    print("Universal Statistical Simulator Demonstration")
    print("Based on: Universal Statistical Simulator by Mark Carney and Ben Varcoe")
    print("=" * 70)
    
    # Test different bias parameters
    biases = [0.3, 0.5, 0.7]
    levels = [6, 8, 10]
    
    for bias in biases:
        for num_levels in levels:
            print(f"\n{'='*20} Bias: {bias}, Levels: {num_levels} {'='*20}")
            
            galton_board = AdvancedQuantumGaltonBoard(num_levels=num_levels, bias=bias)
            results = galton_board.run_comprehensive_experiment(shots=1000, plot=True)
            
            # Print key statistics
            stats = results['stats']
            theoretical_mean = num_levels * bias
            theoretical_std = np.sqrt(num_levels * bias * (1 - bias))
            
            print(f"Experimental Mean: {stats['mean']:.3f} (Theoretical: {theoretical_mean:.3f})")
            print(f"Experimental Std: {stats['std_dev']:.3f} (Theoretical: {theoretical_std:.3f})")
            print(f"Graph saved to output folder")

def main():
    """
    Main function to run advanced implementation with command line arguments
    """
    parser = argparse.ArgumentParser(description='Advanced Quantum Galton Board Implementation')
    parser.add_argument('--levels', '-n', type=int, default=8, 
                       help='Number of levels in the Galton Board (default: 8)')
    parser.add_argument('--bias', '-b', type=float, default=0.5,
                       help='Bias parameter (0.0-1.0, default: 0.5)')
    parser.add_argument('--shots', '-s', type=int, default=1000,
                       help='Number of shots for simulation (default: 1000)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with multiple parameters')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting (only save to file)')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demonstration
        demonstrate_universal_simulator()
    else:
        # Run single experiment
        print("Advanced Quantum Galton Board Implementation")
        print("Based on: Universal Statistical Simulator by Mark Carney and Ben Varcoe")
        print("=" * 70)
        print(f"Parameters: Levels={args.levels}, Bias={args.bias}, Shots={args.shots}")
        print("=" * 70)
        
        galton_board = AdvancedQuantumGaltonBoard(num_levels=args.levels, bias=args.bias)
        galton_board.run_comprehensive_experiment(shots=args.shots, plot=not args.no_plot)
        print(f"Graph saved to output folder")

if __name__ == "__main__":
    main() 
