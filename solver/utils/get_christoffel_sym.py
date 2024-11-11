import numpy as np


# TODO: Test
# TODO: cythonize


def get_christoffel_sym(inverse_tensor: np.ndarray[np.float64], diff_1_inv: np.ndarray[np.float64],
                        i: int, k: int, l: int) -> np.ndarray[np.float64]:
    gamma: np.float64 = 0

    for m in range(4):
        gamma += 1/2 * inverse_tensor[i, m] * (diff_1_inv[m, k, l] + diff_1_inv[m, l, k] - diff_1_inv[k, l, m])
    return gamma
