# Quantum Galton Board Implementation

This repository contains an implementation of the Quantum Galton Board based on the paper **"Universal Statistical Simulator"** by Mark Carney and Ben Varcoe (arXiv:2202.01735).

## Overview

The Quantum Galton Board is a quantum computing implementation that simulates the classical Galton Board (also known as the bean machine or quincunx) using quantum circuits. The classical Galton Board demonstrates the central limit theorem by showing how balls falling through a series of pegs create a binomial distribution.

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

### Files
- `quantum_galton_board.py`: Basic implementation with simple quantum circuit
- `advanced_quantum_galton.py`: Advanced implementation following the paper more closely
- `requirements.txt`: Required Python packages

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
