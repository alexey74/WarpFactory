import numpy as np

from Solver import take_finite_diff_1dir
from Solver.utils.get_christoffel_sym import get_christoffel_sym


# TODO: Test


def cov_div(input_tensor, inverse_tensor, vec_u, vec_d, idx_div, idx_vec, delta, stair_sel):

    diff_1: np.ndarray[np.float64] = np.zeros((4, 4, 4))

    size = input_tensor.shape[2:]

    phi_flag: bool = False

    for i in range(4):
        for j in range(4):
            if i == 2 and j == 2 and size[1] == 1:
                phi_flag = True
            for k in range(4):
                diff_1[i, j, k] = take_finite_diff_1dir(input_tensor[i, j], k, delta, phi_flag)

    # Covariant derivative of covariant vector
    if stair_sel == 0:
        # Build gradient operated vector
        cd_vec = take_finite_diff_1dir(vec_d[idx_vec], idx_div, delta, False)
        for i in range(4):
            gamma = get_christoffel_sym(inverse_tensor, diff_1, i, idx_vec, idx_div)
            cd_vec -= (gamma * vec_d[i])
        return cd_vec
    # covariant derivative of contravariant vector
    elif stair_sel == 1:
        # Build gradient operated vector
        cd_vec = take_finite_diff_1dir(vec_u[idx_vec], idx_div, delta, False)
        for i in range(4):
            gamma = get_christoffel_sym(inverse_tensor, diff_1, idx_vec, idx_div, i)
            cd_vec += (gamma * vec_u[i])
        return cd_vec
    else:
        raise ValueError('Invalid variance selected')
