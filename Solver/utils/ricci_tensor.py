# For some reason the resulting tensor is not the same as the matlab one and it is not explainable with just rounding errors
import numpy as np

from scipy.constants import c
from Solver import take_finite_diff_1dir, take_finite_diff_2dirs


def ricci_tensor(input_tensor: np.ndarray[np.float64], inverse_tensor: np.ndarray[np.float64],
                   delta: np.ndarray[np.float64], use_gpu: bool = False, dim: int = 4) -> np.ndarray[np.float64]:
    phi_flag: bool = False

    size: tuple[int, ...] = input_tensor.shape
    dim: int = input_tensor.shape[0]

    ricci_t: np.ndarray[np.float64] = np.zeros(size)

    diff_1: np.ndarray[np.float64] = np.zeros((dim,) + size)
    diff_2: np.ndarray[np.float64] = np.zeros((dim, dim) + size)

    for i in range(dim):
        for j in range(i, dim):
            for k in range(dim):
                diff_1[i, j, k] = take_finite_diff_1dir(input_tensor[i, j], k, delta, phi_flag)

                if k == 0:
                    diff_1[i, j, k] *= 1/c

                for n in range(k, dim):
                    diff_2[i, j, k, n] = take_finite_diff_2dirs(input_tensor[i, j], k, n, delta, phi_flag)

                    if (n == 0 and k != 0) or (n != 0 and k == 0):
                        diff_2[i, j, k, n] *= 1/c
                    elif n == 0 and k == 0:
                        diff_2[i, j, k, n] *= 1/c**2

                    if k != n:
                        diff_2[i, j, n, k] = diff_2[i, j, k, n]

    for k in range(dim):
        diff_1[1, 0, k] = diff_1[0, 1, k]
        diff_1[2, 0, k] = diff_1[0, 2, k]
        diff_1[2, 1, k] = diff_1[1, 2, k]
        diff_2[3, 0, k] = diff_2[0, 3, k]
        diff_2[3, 1, k] = diff_2[1, 3, k]
        diff_2[3, 2, k] = diff_2[2, 3, k]
        for n in range(dim):
            diff_2[1, 0, k, n] = diff_2[0, 1, k, n]
            diff_2[2, 0, k, n] = diff_2[0, 2, k, n]
            diff_2[2, 1, k, n] = diff_2[1, 2, k, n]
            diff_2[3, 0, k, n] = diff_2[0, 3, k, n]
            diff_2[3, 1, k, n] = diff_2[1, 3, k, n]
            diff_2[3, 2, k, n] = diff_2[2, 3, k, n]

    for i in range(dim):
        for j in range(i, dim):
            r_temp_1: np.ndarray[np.float64] = np.zeros(size[2:])

            for a in range(dim):
                for b in range(dim):
                    r_temp_2: np.ndarray[np.float64] = np.zeros(size[2:])

                    r_temp_2 -= (diff_2[i, j, a, b] + diff_2[a, b, i, j] - diff_2[i, b, j, a] - diff_2[j, b, i, a])

                    for r in range(dim):
                        r_temp_3: np.ndarray[np.float64] = np.zeros(size[2:])
                        r_temp_4: np.ndarray[np.float64] = np.zeros(size[2:])
                        r_temp_5: np.ndarray[np.float64] = np.zeros(size[2:])

                        for d in range(dim):
                            # Second term
                            r_temp_3 += (diff_1[b, d, j] * inverse_tensor[r, d])
                            r_temp_4 += (diff_1[j, d, b] - diff_1[j, b, d]) * inverse_tensor[r, d]
                            # Third term
                            r_temp_5 -= (diff_1[b, d, a] + diff_1[b, d, a] - diff_1[a, b, d]) * inverse_tensor[r, d]
                        r_temp_2 += ((r_temp_4 * diff_1[i, r, a]) + 0.5 * (r_temp_3 * diff_1[a, r, i] +
                                            r_temp_5 * (diff_1[j, r, i] + diff_1[i, r, j] - diff_1[j, i, r])))
                    r_temp_1 += (inverse_tensor[a, b] * r_temp_2)
            ricci_t[i, j] = (0.5 * r_temp_1)
    # Assign symmetric values
    ricci_t[1, 0] = ricci_t[0, 1]
    ricci_t[2, 0] = ricci_t[0, 2]
    ricci_t[2, 1] = ricci_t[1, 2]
    ricci_t[3, 0] = ricci_t[0, 3]
    ricci_t[3, 1] = ricci_t[1, 3]
    ricci_t[3, 2] = ricci_t[2, 3]

    return ricci_t
