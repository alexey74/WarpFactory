import numpy as np


def legendre_radial_interp(input_array, r):
    # 3rd Order Legendre Polynomial Interpolation
    r_scale = 1

    x0 = np.int32(np.max(np.floor(r / r_scale - 1)))
    x1 = np.int32(np.max(np.floor(r / r_scale)))
    x2 = np.int32(np.max(np.ceil(r / r_scale)))
    x3 = np.int32(np.max(np.ceil(r / r_scale + 1)))


    y0 = input_array[x0]
    y1 = input_array[x1]
    y2 = input_array[x2]
    y3 = input_array[x3]

    x0 *= r_scale
    x1 *= r_scale
    x2 *= r_scale
    x3 *= r_scale

    return (((y0 * (r - x1) * (r - x2) * (r - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3))
              + y1 * (r - x0) * (r - x2) * (r - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3)))
             + y2 * (r - x0) * (r - x1) * (r - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3)))
            + y3 * (r - x0) * (r - x1) * (r - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2)))
