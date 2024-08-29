import numpy as np


def shape_func_alcubierre(r: np.float64, R: np.float64, sigma: np.float64) -> np.float64:
    return (np.tanh(sigma * (R + r)) + np.tanh(sigma * (R - r))) / (2 * np.tanh(R * sigma))
