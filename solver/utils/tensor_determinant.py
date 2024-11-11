"""c_det
This function treats the first two dimensions as a 2D Tensor (matrix) of 4D Tensors."""
import numpy as np


def _2d_det(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return tensor[0, 0] * tensor[1, 1] - tensor[0, 1] * tensor[1, 0]


def _3d_det(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    size: tuple[int, ...] = tensor.shape
    ret_tensor: np.ndarray[np.float64] = np.zeros(size[2:])

    for i in range(size[0]):
        index_negative = list(range(size[0]))
        index_negative.remove(i)

        ret_tensor += (-1)**(i + 1) * tensor[0, i] * _2d_det(tensor[1:][:, index_negative])
    return ret_tensor


def _4d_det(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    size: tuple[int, ...] = tensor.shape
    ret_tensor: np.ndarray[np.float64] = np.zeros(size[2:])

    for i in range(size[0]):
        index_negative = list(range(size[0]))
        index_negative.remove(i)

        ret_tensor += (-1)**(i + 1) * tensor[0, i] * _3d_det(tensor[1:][:, index_negative])
    return ret_tensor


def _5d_det(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    size: tuple[int, ...] = tensor.shape
    ret_tensor: np.ndarray[np.float64] = np.zeros(size[2:])

    for i in range(size[0]):
        index_negative = list(range(size[0]))
        index_negative.remove(i)

        ret_tensor += (-1)**(i + 1) * tensor[0, i] * _4d_det(tensor[1:][:, index_negative])
    return ret_tensor


def tensor_determinant(tensor: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    var_err: str = "The tensor must be symmetric in the first 2 of its 6 Dimensions."

    size: tuple = tensor.shape
    n_tensor: np.ndarray[np.float64] = np.zeros(size[2:6])

    h, w = size[:2]
    if h != w or tensor.ndim < 6 or not 1 < h < 6:
        raise ValueError(var_err)

    if h == 2:
        n_tensor = _2d_det(tensor)
    elif h == 3:
        n_tensor = _3d_det(tensor)
    elif h == 4:
        n_tensor = _4d_det(tensor)
    elif h == 5:
        n_tensor = _5d_det(tensor)
    return n_tensor
