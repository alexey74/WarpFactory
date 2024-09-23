"""c_inv
This function treats the first two dimensions as a 2D Tensor (matrix) of 4D Tensors."""
import numpy as np

from Solver.utils.tensor_determinant import tensor_determinant, _3d_det, _2d_det


def _3d_inverse(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    size: tuple[int, ...] = tensor.shape
    minor: np.ndarray[np.float64] = np.zeros(size)

    inv_det: np.ndarray[np.float64] = 1/tensor_determinant(tensor)

    for i in range(size[0]):
        index_negative_first = list(range(size[0]))
        index_negative_first.remove(i)
        for j in range(size[1]):
            index_negative_second = list(range(size[1]))
            index_negative_second.remove(j)

            minor[i, j] = (-1)**(i + j + 1) * inv_det * _2d_det(tensor[index_negative_first][:, index_negative_second])
    return minor.transpose([1, 0, 2, 3, 4, 5])


def _4d_inverse(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    size: tuple[int, ...] = tensor.shape
    minor: np.ndarray[np.float64] = np.zeros(size)

    inv_det: np.ndarray[np.float64] = 1/tensor_determinant(tensor)

    for i in range(size[0]):
        index_negative_first = list(range(size[0]))
        index_negative_first.remove(i)
        for j in range(size[1]):
            index_negative_second = list(range(size[1]))
            index_negative_second.remove(j)

            minor[i, j] = (-1)**(i + j + 1) * inv_det * _3d_det(tensor[index_negative_first][:, index_negative_second])
    return minor.transpose([1, 0, 2, 3, 4, 5])


def tensor_inverse(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    var_err: str = "The tensor must be symmetric in the first 2 of its 6 Dimensions."

    size: tuple = tensor.shape
    n_tensor: np.ndarray[np.float64] = np.zeros(size[2:6])

    h, w = size[:2]
    if h != w or tensor.ndim < 6 or not 1 < h < 6:
        raise ValueError(var_err)

    if h == 3:
        n_tensor = _3d_inverse(tensor)
    elif h == 4:
        n_tensor = _4d_inverse(tensor)
    return n_tensor
