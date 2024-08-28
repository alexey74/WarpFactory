"""
SETMINKOWSKI: Returns the 3+1 format for flat space

INPUTS:
gridSize - World size in [t,x,y,z]

OUTPUTS:
alpha - Lapse rate 4D array

beta - Shift vector, 1x3 cell of 4D arrays

gamma - Spatial terms, 3x3 cell of 4D arrays.
"""
import numpy as np


def set_minkowski_three_plus_one(grid_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_grid_size: tuple = tuple(grid_size)
    alpha: np.ndarray = np.ones(t_grid_size)

    beta = np.zeros((3,) + t_grid_size)

    gamma = np.zeros((3, 3) + t_grid_size)

    for i in range(3):
        for j in range(3):
            if i == j:
                gamma[i, j] = np.ones(t_grid_size)

    return alpha, beta, gamma
