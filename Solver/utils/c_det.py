# c_det Finds the determinant of a cell array
import numpy as np


def c_det(tensor: np.ndarray) -> np.ndarray:
    if tensor.shape[:1][0] != 4 or tensor.shape[1:2][0] != 4:
        if tensor.shape[:1][0] == 3 or tensor.shape[1:2][0] == 3:
            return (tensor[0, 0] * tensor[1, 1] * tensor[2, 2] - tensor[0, 0] * tensor[1, 2] * tensor[2, 1] -
                    tensor[0, 1] * tensor[1, 0] * tensor[2, 2] + tensor[0, 1] * tensor[1, 2] * tensor[2, 0] +
                    tensor[0, 2] * tensor[1, 0] * tensor[2, 1] - tensor[0, 2] * tensor[1, 1] * tensor[2, 0])
        raise ValueError("The tensor must be 4x4 or 3x3 in the first 2 Dimensions.")
    det: np.ndarray = np.zeros(tensor.shape[2:])
    dim: np.int32 = det.shape[:1][0]

    for i in range(dim):
        sub_tensor = np.delete(np.delete(tensor, 0, 0), i, 1)

        sub_det: np.ndarray = np.zeros(sub_tensor.shape[2:])
        for j in range(dim - 1):
            matrix = np.delete(np.delete(sub_tensor, 0, 0), j, 1)
            matrix = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
            sub_det += (2 * np.mod(j, 2) - 1) * sub_tensor[0, j] * matrix
        det += (2 * np.mod(i, 2) - 1) * tensor[0, i] * sub_det
    return det
