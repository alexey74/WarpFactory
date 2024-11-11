# ref: https://en.wikipedia.org/wiki/Trilinear_interpolation
import numpy as np


def trilinear_interpolation(input_tensor: np.ndarray[np.float64], x: np.ndarray[np.float64]):

    x += 10**(-8)

    fl_0: np.int64 = np.floor(x[0]).astype(np.int64) - 1
    fl_1: np.int64 = np.floor(x[1]).astype(np.int64) - 1
    fl_2: np.int64 = np.floor(x[2]).astype(np.int64) - 1

    ce_0: np.int64 = np.ceil(x[0]).astype(np.int64) - 1
    ce_1: np.int64 = np.ceil(x[1]).astype(np.int64) - 1
    ce_2: np.int64 = np.ceil(x[2]).astype(np.int64) - 1

    x_d: np.float64 = (x[0] - np.floor(x[0])) / (np.ceil(x[0]) - np.floor(x[0]))
    y_d: np.float64 = (x[1] - np.floor(x[1])) / (np.ceil(x[1]) - np.floor(x[1]))
    z_d: np.float64 = (x[2] - np.floor(x[2])) / (np.ceil(x[2]) - np.floor(x[2]))

    return ((((((input_tensor[fl_0, fl_1, fl_2] * (1 - x_d)) + (input_tensor[ce_0, fl_1, fl_2] * x_d)) * (1 - y_d)) +
              (((input_tensor[fl_0, ce_1, fl_2] * (1 - x_d)) + (input_tensor[ce_0, ce_1, fl_2] * x_d)) * y_d)) * (1-z_d)) +
            (((((input_tensor[fl_0, fl_1, ce_2] * (1 - x_d)) + (input_tensor[ce_0, fl_1, ce_2] * x_d)) * (1 - y_d)) +
              (((input_tensor[fl_0, ce_1, ce_2] * (1 - x_d)) + (input_tensor[ce_0, ce_1, ce_2] * x_d)) * y_d)) * z_d))

