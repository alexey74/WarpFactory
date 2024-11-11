# TODO: Test
import numpy as np

from analyzer import trace, change_tensor_index
from metrics import Metric, three_plus_one_decomposer
from solver import tensor_inverse
from solver.utils.cov_div import cov_div


def scalar(metric_val: Metric):
    alpha, beta_up = three_plus_one_decomposer(metric_val)[:1]

    size: tuple = metric_val.tensor.shape[2:]

    u_up: np.ndarray[np.float64] = np.zeros(size + (4,))
    u_down: np.ndarray[np.float64] = np.zeros(size + (4,))

    for t in range(size[0]):
        for i in range(size[1]):
            for j in range(size[2]):
                for k in range(size[3]):
                    u_up[t, i, j, k] = 1 / alpha[t, i, j, k] * [1, -beta_up[t, i, j, k, 0], -beta_up[t, i, j, k, 1],
                                                                -beta_up[t, i, j, k, 2]]
                    u_down[t, i, j, k] = np.squeeze(metric_val.tensor[t, i, j, k]) * np.squeeze(u_up[t, i, j, k])

    del_u: np.ndarray[np.float64] = np.zeros((4, 4) + size)
    for i in range(4):
        for j in range(4):
            del_u[i, j] = cov_div(metric_val.tensor, tensor_inverse(metric_val.tensor), u_up, u_down, i, j,
                                  np.array([1, 1, 1, 1]), 0)

    p_mix: np.ndarray[np.float64] = np.zeros((4, 4))
    p: np.ndarray[np.float64] = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            k_delta: int = 0
            if i == j:
                k_delta = 1
            p_mix[i, j] = k_delta + u_up[i] * u_down[j]
            p[i, j] = metric_val.tensor[i, j] + u_down[i] * u_down[j]

    # Define theta tensor
    expansion_tensor: Metric = Metric("Expansion Tensor")
    expansion_tensor.tensor = np.zeros((4, 4) + size)
    expansion_tensor.index = "covariant"
    expansion_tensor.type = "tensor"

    # Define omega tensor
    vorticity_tensor: Metric = Metric("Vorticity Tensor")
    vorticity_tensor.tensor = np.zeros((4, 4) + size)
    vorticity_tensor.index = "covariant"
    vorticity_tensor.type = "tensor"

    for i in range(4):
        for j in range(4):
            for a in range(4):
                for b in range(4):
                    expansion_tensor.tensor[i, j] += p_mix[a, i] * p_mix[b, j] * 1/2 * (del_u[a, b] + del_u[b, a])
                    vorticity_tensor.tensor[i, j] += p_mix[a, i] * p_mix[b, j] * 1 / 2 * (del_u[a, b] - del_u[b, a])

    # Get the trace of theta to calculate its scalar
    expansion_scalar: np.ndarray[np.float64] = trace(expansion_tensor, metric_val)

    # Calculate omega scalar
    vorticity_tensor_up: np.ndarray[np.float64] = change_tensor_index(vorticity_tensor, "contravariant", metric_val)
    vorticity_scalar: np.ndarray[np.float64] = np.zeros(size)
    for i in range(4):
        for j in range(4):
            vorticity_scalar += 1/2 * vorticity_tensor_up[i, j] * vorticity_tensor.tensor[i, j]

    # Shear
    # Define Shear Tensor
    shear_tensor: Metric = Metric("Shear Tensor")
    shear_tensor.tensor = np.zeros((4, 4) + size)
    shear_tensor.index = "covariant"
    shear_tensor.type = "tensor"
    for i in range(4):
        for j in range(4):
            shear_tensor.tensor[i, j] = expansion_tensor.tensor[i, j] - expansion_scalar * 1/3 * p[i, j]

    # Calculate scalar shear
    shear_up: np.ndarray[np.float64] = change_tensor_index(shear_tensor, "contravariant", metric_val)
    shear_scalar: np.ndarray[np.float64] = np.zeros(size)
    for i in range(4):
        for j in range(4):
            shear_scalar += 1/2 * shear_tensor.tensor[i, j] * shear_up[i, j]
    return shear_scalar, expansion_scalar, vorticity_scalar

