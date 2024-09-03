import numpy as np


def c3_inv(tensor: np.ndarray) -> np.ndarray:
    h, w = tensor.shape[:2]
    assert h == 3 and w == 3, 'Tensor is not 3x3'

    inv_det = (tensor[0, 0] * tensor[1, 1] * tensor[2, 2] - tensor[0, 0] * tensor[1, 2] * tensor[2, 1] -
           tensor[0, 1] * tensor[1, 0] * tensor[2, 2] + tensor[0, 1] * tensor[1, 2] * tensor[2, 0] +
           tensor[0, 2] * tensor[1, 0] * tensor[2, 1] - tensor[0, 2] * tensor[1, 1] * tensor[2, 0])
    return np.array([[inv_det * (tensor[1, 1] * tensor[2, 2] - tensor[1, 2] * tensor[2, 1]),  inv_det *
                      (tensor[0, 2] * tensor[2, 1] - tensor[0, 1] * tensor[2, 2]),  inv_det *
                      (tensor[0, 1] * tensor[1, 2] - tensor[0, 2] * tensor[1, 1])],
                     [inv_det * (tensor[1, 2] * tensor[2, 0] - tensor[1, 0] * tensor[2, 2]),  inv_det *
                      (tensor[0, 0] * tensor[2, 2] - tensor[0, 2] * tensor[2, 0]),  inv_det *
                      (tensor[0, 2] * tensor[1, 0] - tensor[0, 0] * tensor[1, 2])],
                     [inv_det * (tensor[1, 0] * tensor[2, 1] - tensor[1, 1] * tensor[2, 0]),  inv_det *
                      (tensor[0, 1] * tensor[2, 0] - tensor[0, 0] * tensor[2, 1]),  inv_det *
                      (tensor[0, 0] * tensor[1, 1] - tensor[0, 1] * tensor[1, 0])]])
