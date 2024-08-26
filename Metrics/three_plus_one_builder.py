"""
THREEPLUSONEBUILDER: Builds the metric given input 3+1 components of alpha, beta, and gamma

INPUTS:
alpha - (TxXxYxZ) lapse rate map across spacetime
beta - {3}x(TxXxYxZ) (covariant assumed) shift vector map across spacetime
gamma - {3x3}x(TxXxYxZ) (covariant assumed) spatial term map across spacetime


OUTPUTS:
metricTensor - metric struct
"""
import numpy as np


def three_plus_one_builder(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:

    # Set spatial components
    h_gamma, w_gamma = gamma.shape[:2]
    assert h_gamma == 3 and w_gamma == 3, 'Cell array is not 3x3'
    gamma_up = np.linalg.inv(gamma)

    # Find gridSize
    s = tuple(gamma.shape[2:])

    # Calculate beta_i
    beta_up = np.zeros((1, 3) + s)

    for i in range(3):
        for j in range(3):
            beta_up[0, i] = beta_up[0, i] + gamma_up[i, j] * beta[0, j]

    # Create time-time component
    metric_tensor: np.ndarray = np.zeros((4, 4) + s)
    metric_tensor[0, 0] = -alpha**2

    for i in range(3):
        metric_tensor[0, 0] = metric_tensor[0, 0] + beta_up[0, i] * beta[0, i]

    # Create time-space components
    for i in range(1, 4):
        metric_tensor[0, i] = beta[0, i-1]
        metric_tensor[i, 0] = metric_tensor[1, i]

    # Create space-space components
    for i in range(1, 4):
        for j in range(1, 4):
            metric_tensor[i, j] = gamma[i-1, j-1]

    return metric_tensor
