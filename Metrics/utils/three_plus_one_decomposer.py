"""
THREEPLUSONEDECOMPOSER: Finds 3+1 terms from the metric tensor

    INPUTS:
    metric - metric struct object.

    OUTPUTS:
    alpha - 4D array. Lapse rate.

    betaDown - 1x3 cell of 4D arrays. Shift vectors.

    gamma - 3x3 cell of 4D arrays. Spatial terms.
"""
import numpy as np

from Analyzer.change_tensor_index import change_tensor_index
from Solver import tensor_inverse


def three_plus_one_decomposer(metric_val) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Check that the metric is covariant and change index if not
    metric_val = change_tensor_index(metric_val, "covariant")

    # Covariant shift vector maps to the covariant tensor terms g_0i
    beta_down: np.ndarray = np.array([[metric_val.tensor[0, 1], metric_val.tensor[1, 2], metric_val.tensor[2, 3]]])

    # Covariant gamma maps to the covariant tensor terms g_ij
    gamma_down: np.ndarray = np.array([[metric_val.tensor[1, 1], metric_val.tensor[1, 2], metric_val.tensor[1, 3]],
                                       [metric_val.tensor[2, 1], metric_val.tensor[2, 2], metric_val.tensor[2, 3]],
                                       [metric_val.tensor[3, 1], metric_val.tensor[3, 2], metric_val.tensor[3, 3]]])

    # Set spatial components
    gamma_up: np.ndarray = tensor_inverse(gamma_down)

    # Transform beta to contravariant
    beta_up: np.ndarray = np.zeros((1, 3) + metric_val.tensor[0, 0].shape)

    for i in range(3):
        for j in range(3):
            beta_up[0, i] = beta_up[0, i] + gamma_up[i, j] * beta_down[0, j]

    # Find lapse using beta and g_00
    alpha: np.ndarray = np.sqrt(beta_up[0, 0] * beta_down[0, 0] + beta_up[0, 1] * beta_down[0, 1] +
                                      beta_up[0, 2] * beta_down[0, 2] - metric_val.tensor[0, 0])

    return alpha, beta_up, beta_down, gamma_up, gamma_down
