"""
THREEPLUSONEBUILDER: Builds the metric given input 3+1 components of alpha, beta, and gamma

    INPUTS:
    alpha - (TxXxYxZ) lapse rate map across spacetime
    beta - {3}x(TxXxYxZ) (covariant assumed) shift vector map across spacetime
    gamma - {3x3}x(TxXxYxZ) (covariant assumed) spatial term map across spacetime


    OUTPUTS:
    tensor - metric struct
"""
import numpy as np

from solver import tensor_inverse


def three_plus_one_builder(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    # Set spatial components
    gamma_up = tensor_inverse(gamma)

    # Find gridSize
    s = gamma.shape[2:]

    # Calculate beta_i
    beta_up = np.zeros((3,) + s)

    for i in range(3):
        for j in range(3):
            beta_up[i] += gamma_up[i, j] * beta[j]

    # Create time-time component
    tensor: np.ndarray = np.zeros((4, 4) + s)
    tensor[0, 0] = -alpha**2

    for i in range(3):
        tensor[0, 0] += beta_up[i] * beta[i]

    # Create time-space components
    for i in range(1, 4):
        tensor[0, i] = beta[i-1]
        tensor[i, 0] = tensor[0, i]

    # Create space-space components
    for i in range(1, 4):
        for j in range(1, 4):
            tensor[i, j] = gamma[i-1, j-1]

    return tensor
