# c_det Finds the determinant of a cell array
import numpy as np


def c_det(tensor: np.ndarray) -> np.float64:
    det: np.float64 = np.float64(0.0)

    h, w = tensor.shape[:2]
    assert h == 4 and w == 4, 'Tensor is not 4x4'

    if h == 2 and w == 2:
        return tensor[0, 0] * tensor[1, 1] - tensor[0, 1] * tensor[1, 0]

    for i in range(h):
        sub_tensor = tensor
        sub_tensor[1:] = []
        sub_tensor[:1] = []
        sub_det = c_det(sub_tensor)
        det += (2 * np.mod(i, 2) - 1) * tensor[1, i] * sub_det
    return det