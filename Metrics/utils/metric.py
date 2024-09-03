import numpy as np

from dataclasses import dataclass

@dataclass
class Metric:
    name: str
    type: str = None
    frame: str = None
    tensor: np.ndarray[np.float64] = None
    scaling: np.ndarray[np.float64] = None
    coords: str = None
    index: str = None
    date: str = None
    params_p: np.float64 = None
    params_rho_smooth: np.ndarray[np.float64] = None
    params_world_center: np.ndarray[np.float64] = None
    params_grid_size: np.ndarray[np.float64] = None
    params_velocity: np.float64 = None
    params_rs: np.float64 = None
    params_big_r: np.float64 = None
    params_sigma: np.float64 = None
    params_big_r_1: np.float64 = None
    params_sigma_1: np.float64 = None
    params_big_r_2: np.float64 = None
    params_sigma_2: np.float64 = None
    params_big_a: np.float64 = None
    params_big_b: np.float64 = None