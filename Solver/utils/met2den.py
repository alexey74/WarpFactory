"""
GETENERGYTENSOR: Converts the metric into the stress energy tensor

INPUTS:
metric - A metric struct

tryGPU - A flag on whether or not to use GPU computation (0=no, 1=yes)

diffOrder - Order of finite difference, either 'second' or 'fourth'

OUTPUTS:
energy - energy tensor struct
"""
import numpy as np


from Solver import tensor_inverse, ricci_tensor, ricci_scalar, einstein_tensor, energy_density


# TODO: Implement actual GPU Support


def met2den(input_tensor: np.ndarray[np.float64], delta: np.ndarray[np.float64] = np.array([1, 1, 1, 1]),
            order: int = 4, use_gpu: bool = False) -> np.ndarray[np.float64]:
    inverse_tensor: np.ndarray = tensor_inverse(input_tensor)

    ricci_t: np.ndarray[np.float64] = ricci_tensor(input_tensor, inverse_tensor, delta, use_gpu, order)

    ricci_s: np.float64 = ricci_scalar(inverse_tensor, ricci_t, use_gpu)

    return energy_density(einstein_tensor(input_tensor, ricci_t, ricci_s, use_gpu), inverse_tensor, use_gpu)
