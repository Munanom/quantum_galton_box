# Project Name
Quantum Walks and Monte Carlo

# Team Name
PhiZero

# Team Members
Muna Numan Said - gst-p2RT7Pdi4yyDWD2 <br/>
Aratrika Gupta - gst-oR0co0qYzvHNIQu <br/>
Kashinath Gokarn - 

# 500 word Summary

## Overview

This project extended a 1- and 2-layer Quantum Galton Board to a generalized algorithm for any number of layers, implemented using Qiskit on the `FakeMontrealV2` backend, a 27-qubit emulator with realistic noise. The objectives were to simulate binomial, exponential, and Hadamard quantum walk distributions, verify Gaussian convergence for the binomial case, optimize performance under noise, and compute distance metrics with stochastic uncertainty.

## Generalized Algorithm and Gaussian Verification

The `QuantumGaltonBoard` class was developed to create quantum circuits for an arbitrary number of layers (`num_levels`). Each qubit undergoes a Hadamard gate, simulating a binomial distribution (`p=0.5`), with measurement outcomes (number of 1s) representing positions. Using `qiskit-aer`’s `qasm_simulator`, we ran experiments with 8 levels and 1000 shots, both noiseless and with `FakeMontrealV2` noise. The noiseless simulation yielded a binomial distribution with a mean of 3.94 (theoretical: 4.00) and standard deviation of 1.44 (theoretical: 1.41). The `test_gaussian_convergence` method tested levels [4, 8, 12, 16], confirming Gaussian convergence per the central limit theorem. Plots saved to the `output/` folder showed experimental histograms aligning closely with theoretical binomial and Gaussian distributions (mean `n*0.5`, standard deviation `sqrt(n*0.25)`).

## Modified Distributions

The `AdvancedQuantumGaltonBoard` class extended the framework to simulate exponential and Hadamard quantum walk distributions using a noiseless all-to-all sampler (`qasm_simulator`). For the exponential distribution, we used Qiskit’s `StatePreparation` to encode probabilities with `lambda=1.5` and 6 precision bits, achieving an exponential decay profile. The Hadamard quantum walk employed a coin qubit and 5 position qubits, with optimized multi-controlled gates for shifting, simulating 6 steps. Both distributions matched theoretical expectations: exponential decay and a symmetric, Gaussian-like spread, respectively, as visualized in output plots.

## Noisy Hardware Optimization

To maximize accuracy under noise, we used the `FakeMontrealV2` noise model, simulating IBM quantum hardware errors. The binomial circuit was optimized by minimizing gate depth (Hadamard gates only) and tested up to 8 levels. The exponential circuit used efficient state preparation, and the Hadamard walk reduced gate count via multi-controlled gates. Despite noise, the binomial simulation achieved excellent distance metrics: KL divergence (0.0045 ± 0.0022), JS divergence (0.0012 ± 0.0007), and total variation (0.0223 ± 0.0081). Measurement mitigation (`CompleteMeasFitter`) was attempted but disabled due to missing `qiskit-experiments`, yet results remained robust.

## Distance Metrics and Uncertainty

The `utils.py` module computed KL divergence, JS divergence, and total variation between experimental and theoretical distributions, with stochastic uncertainty estimated over 5 runs. A critical fix in `utils.py` handled `FakeMontrealV2`’s padded bitstrings by counting 1s in the first `num_levels` bits, ensuring accurate position mapping. Low distance metrics confirmed high fidelity to theoretical distributions, with uncertainties reflecting statistical variations.

## Challenges and Future Work

Challenges included handling padded bitstrings and missing mitigation dependencies. Upgrading to Python 3.10 and installing `qiskit-experiments` are recommended for enhanced mitigation. Future work could scale to more levels, test on real IBM hardware, or further optimize gate sequences for noise resilience.

## Conclusion

We successfully generalized the Quantum Galton Board, verified Gaussian convergence, implemented diverse distributions, and optimized for noisy hardware. Low distance metrics and comprehensive visualizations (saved to `output/`) demonstrate high accuracy, making this a robust quantum simulation framework.

*Date: August 10, 2025*

# Project Presentation Deck

It is uploaded as the file Quantum Walks and Monte Carlo (1). pdf along with the summary and the Two Page writeup of our project.

# Quantum Galton Board Implementation

This repository implements a Quantum Galton Board and related quantum walk simulations, inspired by the paper *"Universal Statistical Simulator"* by Mark Carney and Ben Varcoe. The project provides tools to simulate and compare quantum and classical random walks, including binomial, exponential, and Hadamard quantum walk distributions, using quantum circuits and classical Monte Carlo methods. The simulations leverage Qiskit for quantum circuit construction and NumPy/Matplotlib for numerical computations and visualizations.


## Overview

The Quantum Galton Board is a quantum computing implementation that simulates the classical Galton Board (also known as the bean machine or quincunx) using quantum circuits. The classical Galton Board demonstrates the central limit theorem by showing how balls falling through a series of pegs create a binomial distribution.

Key functionalities include:\n
- Simulating a Quantum Galton Board with Hadamard gates to model superposition.\n
- Implementing split-step quantum walks on a 1D lattice.\n
- Comparing quantum and classical random walk distributions.\n
- Supporting multiple distribution types (binomial, exponential, Hadamard walk).\n
- Analyzing convergence to Gaussian distributions for large numbers of levels.


## Theoretical Background

### Classical Galton Board
The classical Galton Board consists of:
- A vertical board with pegs arranged in rows
- Balls dropped from the top that bounce off pegs
- Collection bins at the bottom
- The resulting distribution follows a binomial distribution

### Quantum Implementation
The quantum version uses:
- **Qubits** to represent the possible paths
- **Quantum superposition** to explore all possible paths simultaneously
- **Entanglement** to create correlations between different levels
- **Measurement** to collapse the superposition and obtain results

### Key Features from the Paper
1. **Universal Statistical Simulator**: The quantum circuit can simulate various statistical distributions
2. **Quantum Advantage**: Exploits quantum parallelism to explore multiple paths simultaneously
3. **Tunable Parameters**: Can adjust bias and number of levels to create different distributions

## Implementation

### File Structure
- `split_step_quantum_walks.py`: Implements a split-step quantum walk on a 1D lattice with functions for rotation matrices, state initialization, and measurement.\n
- `run_walk_simulation.py`: Simulates and compares quantum and classical random walks, generating plots and computing metrics like entropy and KL divergence.\n
- `advanced_quantum_galton.py`: Implements an advanced Quantum Galton Board supporting binomial, exponential, and Hadamard walk distributions with detailed visualizations.\n
- `main.py`: Main script to run various examples (simple, advanced, comparison, convergence tests) with command-line argument support.\n
- `quantum_galton_board.py`: Implements a basic Quantum Galton Board using Qiskit, with functions for simulation, analysis, and Gaussian convergence testing.\n
- `output/`: Directory where plots are saved (must be created manually if not present).\n
- `requirements.txt`: Lists required Python packages.



### Key Components

#### 1. Quantum Circuit Design
```python
# Create superposition with Hadamard gates
for i in range(self.num_qubits):
    qc.h(i)

# Apply controlled operations for entanglement
for level in range(self.num_levels - 1):
    for qubit in range(self.num_levels - level - 1):
        qc.cx(qubit, qubit + 1)
        qc.rz(np.pi/4, qubit + 1)
        qc.cx(qubit, qubit + 1)
```

#### 2. Bias Control
The implementation allows for adjustable bias parameters:
- `bias = 0.5`: Fair distribution (classical Galton Board)
- `bias < 0.5`: Left-biased distribution
- `bias > 0.5`: Right-biased distribution

#### 3. Statistical Analysis
- Comparison with theoretical binomial distribution
- Error analysis
- Comprehensive visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd womanium_quantum2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from quantum_galton_board import QuantumGaltonBoard

# Create a Galton Board with 8 levels
galton_board = QuantumGaltonBoard(num_levels=8)

# Run simulation with 1000 shots
results = galton_board.run_experiment(shots=1000, plot=True)
```

### Advanced Usage
```python
from advanced_quantum_galton import AdvancedQuantumGaltonBoard

# Create with custom bias
galton_board = AdvancedQuantumGaltonBoard(num_levels=8, bias=0.7)

# Run comprehensive experiment
results = galton_board.run_comprehensive_experiment(shots=10000, plot=True)
```

### Running Examples
```bash
# Run basic implementation
python quantum_galton_board.py

# Run advanced implementation
python advanced_quantum_galton.py
```

## Results and Visualization

The implementation provides comprehensive visualization including:

1. **Experimental Results**: Histogram of quantum simulation results
2. **Theoretical Comparison**: Side-by-side comparison with binomial distribution
3. **Probability Analysis**: Probability distribution comparison
4. **Error Analysis**: Absolute error between quantum and theoretical results
5. **Statistics Summary**: Key statistical measures
6. **Circuit Information**: Details about the quantum circuit used

## Key Features

### 1. Universal Statistical Simulator
- Can simulate various statistical distributions
- Tunable parameters for different scenarios
- Extensible design for other distributions

### 2. Quantum Circuit Optimization
- Efficient circuit design
- Minimal number of gates
- Optimized for current quantum hardware

### 3. Comprehensive Analysis
- Statistical validation
- Error quantification
- Visual comparison with classical theory

### 4. Educational Value
- Demonstrates quantum-classical correspondence
- Shows quantum advantage in statistical simulation
- Provides hands-on experience with quantum circuits

## Mathematical Foundation

### Binomial Distribution
The theoretical distribution follows:
```
P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
```
where:
- `n` = number of levels
- `k` = number of successful outcomes
- `p` = probability of success (bias parameter)

### Quantum State
The quantum circuit creates a superposition state:
```
|ψ⟩ = Σᵢ cᵢ|i⟩
```
where `|i⟩` represents different possible outcomes and `cᵢ` are complex amplitudes.

## Applications

1. **Educational**: Teaching quantum computing concepts
2. **Research**: Exploring quantum-classical correspondence
3. **Statistical Simulation**: Quantum-enhanced Monte Carlo methods
4. **Algorithm Development**: Foundation for more complex quantum algorithms

## Future Extensions

1. **Multi-dimensional Galton Board**: Extension to higher dimensions
2. **Continuous Distributions**: Simulation of continuous probability distributions
3. **Quantum Hardware**: Implementation on real quantum computers
4. **Machine Learning**: Integration with quantum machine learning algorithms

## References

- Carney, M., & Varcoe, B. (2022). Universal Statistical Simulator. arXiv:2202.01735
- Galton, F. (1889). Natural Inheritance. Macmillan and Co.
- Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information. Cambridge University Press.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements, bug fixes, or additional features.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
