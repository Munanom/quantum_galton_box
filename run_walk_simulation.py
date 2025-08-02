import numpy as np
import matplotlib.pyplot as plt
from quantum_galton_board import QuantumGaltonBoard
from split_step_quantum_walk import qw_split, init_psi_from_distribution, measure
from scipy.stats import entropy

def classical_walk(initial_dist, steps=6, p_left=0.5):
    N = len(initial_dist)
    pad = steps * 2
    dist = np.pad(initial_dist, (pad, pad), mode='constant')

    for _ in range(steps):
        new_dist = np.zeros_like(dist)
        for i in range(1, len(dist) - 1):
            new_dist[i - 1] += dist[i] * p_left
            new_dist[i + 1] += dist[i] * (1 - p_left)
        dist = new_dist

    return dist[pad:-pad]

def plot_comparison(x, quantum_dist, classical_dist, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x, quantum_dist, lw=2, color='darkorange', label='Quantum Walk')
    plt.plot(x, classical_dist, lw=2, linestyle='--', color='blue', label='Classical Walk')
    plt.title('Quantum vs Classical Walk Comparison')
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Parameters
levels = 6
shots = 2048
theta1 = np.pi / 4
theta_m = np.pi / 3
theta_p = np.pi / 6
walk_steps_list = [2, 4, 6, 8]

# Initialize Galton board and get classical output
galton = QuantumGaltonBoard(num_levels=levels)
counts = galton.simulate_galton_board(shots)

N = levels
classical_prob = np.zeros(2 * N + 1)

for bitstring, count in counts.items():
    pos = bitstring.count('1')
    classical_prob[N - levels + pos] = count 

classical_prob /= np.sum(classical_prob)
psi0 = init_psi_from_distribution(N, classical_prob)

# Loop over walk step lengths
for steps in walk_steps_list:
    print(f"\n--- Running walk steps = {steps} ---")
    
    # Run quantum walk
    psi_t = qw_split(N, theta1, theta_m, theta_p, psi0, steps=steps)
    final_dist = measure(psi_t[:, :, -1])
    
    # Run classical walk
    classical_result = classical_walk(classical_prob, steps=steps)
    
    # Position axis
    x = np.arange(-N, N + 1)
    
    # Plot comparison
    plot_comparison(
        x,
        final_dist,
        classical_result,
        save_path=f"output/quantum_vs_classical_walk_{steps}steps.pdf"
    )
    
    # Compute metrics
    q_peak = x[np.argmax(final_dist)]
    c_peak = x[np.argmax(classical_result)]
    kl = entropy(final_dist + 1e-12, classical_result + 1e-12)

    print(f"[Steps = {steps}]")
    print(f"Quantum Peak:   {q_peak}")
    print(f"Classical Peak: {c_peak}")
    print(f"Peak Shift:     {q_peak - c_peak}")
    print(f"Entropy (Q):    {entropy(final_dist):.4f}")
    print(f"Entropy (C):    {entropy(classical_result):.4f}")
    print(f"KL(Q || C):     {kl:.6f}")
