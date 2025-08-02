# split_step_quantum_walk.py

import numpy as np

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
