"""
runs a weighted average between two timeslices after a spatial trilinear
interpolation in both slices.
"""
import numpy as np

from solver import trilinear_interpolation


# TODO: Test


def quadrilinear_interpolation(tensor, x):

    t = x[0]
    t_1 = np.floor(t)
    t_2 = np.ceil(t)

    if t_1 == t_2:
        return trilinear_interpolation(np.squeeze(tensor[t_1, :, :, :]), x[1:])
    else:
        c_1 = trilinear_interpolation(np.squeeze(tensor[t_1, :, :, :]), x[1:])
        c_2 = trilinear_interpolation(np.squeeze(tensor[t_1, :, :, :]), x[1:])

        return (c_1 * (t_2 - t) + c_2 * (t - t_1)) / (t_2 - t_1)
