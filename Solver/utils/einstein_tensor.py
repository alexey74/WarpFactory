import numpy as np


def einstein_tensor(input_tensor: np.ndarray[np.float64], ricci_t: np.ndarray[np.float64],
                                       ricci_s: np.ndarray[np.float64], use_gpu: bool) -> np.ndarray[np.float64]:
    e_tensor: np.ndarray[np.float64] = np.zeros((4, 4) + ricci_t.shape[2:])

    for i in range(4):
        for j in range(4):
            e_tensor[i, j] = ricci_t[i, j] - 0.5 * input_tensor[i, j] * ricci_s
    return e_tensor
