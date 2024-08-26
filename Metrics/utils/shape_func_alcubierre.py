import numpy as np


def shape_func_alcubierre(r, R, sigma):
    return (np.tanh(sigma * (R + r)) + np.tanh(sigma * (R - r))) / (2 * np.tanh(R * sigma))
