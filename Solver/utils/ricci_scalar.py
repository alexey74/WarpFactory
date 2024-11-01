import numpy as np


def ricci_scalar(inverse_tensor: np.ndarray[np.float64], ricci_t: np.ndarray[np.float64], use_gpu: bool) -> np.float64:
    ricci_s: np.float64 = 0

    for i in range(4):
        for j in range(4):
            ricci_s += inverse_tensor[i, j] * ricci_t[i, j]
    return ricci_s
