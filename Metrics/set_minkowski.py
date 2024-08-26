"""
SETMINKOWSKI: Builds metric terms for a flat Minkowski space

INPUTS:
gridSize - World size in [t,x,y,z]

metric - Metric struct

OUTPUTS:
tensor - The metric tensor as a 4x4 cell of 4D arrays.

"""
import numpy as np

from Metrics.metric import Metric


def set_minkowski(grid_size: np.ndarray) -> np.ndarray:
    t_grid_size: tuple = tuple(grid_size)

    metric: np.ndarray = np.zeros((4, 4) + t_grid_size)

    metric[0, 0] = np.full(t_grid_size, -1)

    for i in range(1, 4):
        for j in range(1, 4):
            if i == j:
                metric[i, j] = np.ones(t_grid_size)

    return metric
