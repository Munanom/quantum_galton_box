#!/usr/bin/env python3
"""
Simple example demonstrating the Quantum Galton Board implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_galton_board import QuantumGaltonBoard
from advanced_quantum_galton import AdvancedQuantumGaltonBoard
import argparse

def simple_example(levels: int = 6, shots: int = 500):
    """
    Simple example with basic implementation
    """
    print("=== Simple Quantum Galton Board Example ===")
    
    # Create a Galton Board with specified levels
    galton_board = QuantumGaltonBoard(num_levels=levels)
    
    # Run a quick simulation
    print(f"Running simulation with {shots} shots...")
    results = galton_board.run_experiment(shots=shots, plot=True)
    print("Simple example graph saved to output folder")
    
    return results

def convergence_example(levels: list = None, shots: int = 5000):
    """
    Gaussian convergence test example
    """
    print("=== Gaussian Convergence Test Example ===")
    
    if levels is None:
        levels = [4, 8, 12, 16]
    
    # Create a Galton Board and run convergence test
    galton_board = QuantumGaltonBoard(num_levels=8)  # Default level for initialization
    galton_board.test_gaussian_convergence(levels_list=levels, shots=shots)
    print("Convergence test completed and saved to output folder")

def theoretical_example(levels: list = None):
    """
    Theoretical convergence test example
    """
    print("=== Theoretical Convergence Test Example ===")
    
    if levels is None:
        levels = [10, 20, 50, 100]
    
    # Create a Galton Board and run theoretical test
    galton_board = QuantumGaltonBoard(num_levels=8)  # Default level for initialization
    galton_board.test_theoretical_convergence(levels_list=levels)
    print("Theoretical test completed and saved to output folder")

def advanced_example(levels: int = 8, shots: int = 1000):
    """
    Advanced example with bias control
    """
    print("\n=== Advanced Quantum Galton Board Example ===")
    
    # Test different bias values
    biases = [0.3, 0.5, 0.7]
    
    for bias in biases:
        print(f"\nTesting bias = {bias}")
        
        # Create Galton Board with specific bias
        galton_board = AdvancedQuantumGaltonBoard(num_levels=levels, bias=bias)
        
        # Run simulation
        results = galton_board.run_comprehensive_experiment(shots=shots, plot=True)
        
        # Print key statistics
        stats = results['stats']
        theoretical_mean = levels * bias
        print(f"Experimental mean: {stats['mean']:.2f}")
        print(f"Theoretical mean: {theoretical_mean:.2f}")
        print(f"Advanced example graph saved to output folder")

def comparison_example(levels: int = 10, shots: int = 5000):
    """
    Compare quantum vs classical simulation
    """
    print("\n=== Quantum vs Classical Comparison ===")
    
    # Quantum simulation
    galton_board = AdvancedQuantumGaltonBoard(num_levels=levels, bias=0.5)
    quantum_results = galton_board.run_comprehensive_experiment(shots=shots, plot=False)
    
    # Classical simulation (Monte Carlo)
    print("Running classical Monte Carlo simulation...")
    num_trials = shots
    classical_results = []
    
    for _ in range(num_trials):
        # Simulate ball falling through specified levels
        position = sum(np.random.random() < 0.5 for _ in range(levels))
        classical_results.append(position)
    
    # Compare results
    quantum_positions, quantum_freqs, quantum_stats = galton_board.analyze_distribution(quantum_results['counts'])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Quantum results
    ax1.bar(quantum_positions, quantum_freqs, alpha=0.7, color='blue', label='Quantum')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Quantum Simulation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Classical results
    classical_positions, classical_counts = np.unique(classical_results, return_counts=True)
    ax2.bar(classical_positions, classical_counts, alpha=0.7, color='red', label='Classical')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Classical Monte Carlo')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"output/quantum_vs_classical_comparison_{levels}levels_{timestamp}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show() 
    
    # Print statistics
    classical_mean = np.mean(classical_results)
    classical_std = np.std(classical_results)
    
    print(f"\nComparison Statistics:")
    print(f"Quantum  - Mean: {quantum_stats['mean']:.3f}, Std: {quantum_stats['std_dev']:.3f}")
    print(f"Classical - Mean: {classical_mean:.3f}, Std: {classical_std:.3f}")
    print(f"Theoretical - Mean: {levels * 0.5:.3f}, Std: {np.sqrt(levels * 0.25):.3f}")
    print(f"Comparison graph saved to output folder")

def main():
    """
    Main function to run all examples
    """
    parser = argparse.ArgumentParser(description='Quantum Galton Board Examples')
    parser.add_argument('--levels', '-n', type=int, default=8, 
                       help='Number of levels in the Galton Board (default: 8)')
    parser.add_argument('--shots', '-s', type=int, default=1000,
                       help='Number of shots for simulation (default: 1000)')
    parser.add_argument('--simple-only', action='store_true',
                       help='Run only simple example')
    parser.add_argument('--advanced-only', action='store_true',
                       help='Run only advanced example')
    parser.add_argument('--comparison-only', action='store_true',
                       help='Run only comparison example')
    parser.add_argument('--convergence-only', action='store_true',
                       help='Run only convergence test')
    parser.add_argument('--theoretical-only', action='store_true',
                       help='Run only theoretical test')
    parser.add_argument('--convergence-levels', nargs='+', type=int,
                       help='Custom levels for convergence test')
    parser.add_argument('--theoretical-levels', nargs='+', type=int,
                       help='Custom levels for theoretical test')
    
    args = parser.parse_args()
    
    print("Quantum Galton Board Examples")
    print("=" * 50)
    print(f"Parameters: Levels={args.levels}, Shots={args.shots}")
    print("=" * 50)
    
    try:
        if args.simple_only:
            simple_example(args.levels, args.shots)
        elif args.advanced_only:
            advanced_example(args.levels, args.shots)
        elif args.comparison_only:
            comparison_example(args.levels, args.shots)
        elif args.convergence_only:
            convergence_example(args.convergence_levels, args.shots)
        elif args.theoretical_only:
            theoretical_example(args.theoretical_levels)
        else:
            # Run all examples
            simple_example(args.levels, args.shots // 2)
            advanced_example(args.levels, args.shots)
            comparison_example(args.levels, args.shots)
            convergence_example(args.convergence_levels, args.shots)
            theoretical_example(args.theoretical_levels)
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 
