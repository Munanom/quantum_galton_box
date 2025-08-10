import numpy as np

"""
Implements a split-step quantum walk on a one-dimensional lattice. The module defines functions to:
- Create rotation matrices for coin operations with uniform (rotation_1) and position-dependent (rotation_2) angles.
- Initialize a quantum state from a given probability distribution (init_psi_from_distribution).
- Perform matrix multiplication for coin operations (mult).
- Execute the quantum walk (qw_split) by alternating coin operations and position shifts.
- Measure the final probability distribution (measure).
The quantum walk evolves a quantum state over a specified number of steps, applying rotations and shifts to simulate quantum interference.
"""

def rotation_1(N, theta):
    q = 0.5 * theta * np.ones(2*N + 1)
    return np.array([[np.cos(q), -np.sin(q)],
                     [np.sin(q),  np.cos(q)]])

def rotation_2(N, theta_m, theta_p):
    x = np.arange(-N, N+1)
    delta = 0.01
    q = 0.5 * ((theta_p + theta_m)/2 + (theta_p - theta_m)/2 * np.tanh((x + 0.5)/delta))
    return np.array([[np.cos(q), -np.sin(q)],
                     [np.sin(q),  np.cos(q)]])

def init_psi_from_distribution(N, prob_dist):
    psi = np.zeros((2, 2*N + 1), dtype=complex)
    sqrt_probs = np.sqrt(prob_dist)
    psi[0, :] = sqrt_probs / np.linalg.norm(sqrt_probs)  # normalize
    return psi

def mult(coin, psi):
    return np.einsum('ijk,jk->ik', coin, psi)

def qw_split(N, theta, theta_m, theta_p, psi_init, steps):
    r1 = rotation_1(N, theta)
    r2 = rotation_2(N, theta_m, theta_p)
    psi = psi_init.copy()
    psi_t = np.zeros((2, 2*N + 1, steps+1), dtype=complex)
    psi_t[:, :, 0] = psi

    for n in range(1, steps+1):
        psi = mult(r1, psi)
        psi[0] = np.roll(psi[0], 1)   # shift left
        psi = mult(r2, psi)
        psi[1] = np.roll(psi[1], -1)  # shift right
        psi_t[:, :, n] = psi

    return psi_t

def measure(psi):
    return np.abs(psi[0])**2 + np.abs(psi[1])**2
