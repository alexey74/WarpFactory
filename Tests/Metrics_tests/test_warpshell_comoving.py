import numpy as np
import scipy as sp

from Metrics import Metric, warpshell_comoving
from Tests.utils.arr_hash import arr_hash


warpshell_comoving_tensor_hash: str = 'ec9ab0956eed6c124da641dddc2379388503c222f9427016a4bfcb99d76eed40e2434af9d820249b0700138d0c116999e2797d9344dd20b3588b549736bdab8b'


def test_warpshell_comoving():
    grid_size: np.ndarray[np.float64] = np.array([1, 300, 300, 5], dtype=np.int32)
    world_center: np.ndarray[np.float64] = np.array([1.000692285594456e-11, 30.1, 30.1, 0.6])
    m: np.float64 = np.float64(20/(2*sp.constants.G)*sp.constants.c**2*1/3)
    big_r_1: np.float64 = np.float64(10)
    big_r_2: np.float64 = np.float64(20)
    r_buff: np.float64 = np.float64(0.0)
    sigma: np.float64 = np.float64(0.0)
    smooth_factor: np.float64 = np.float64(4000)
    v_warp: np.float64 = np.float64(0.02)
    do_warp: bool = True
    grid_scaling: np.ndarray[np.float64] = np.array([3.335640951981520e-12, 0.2, 0.2, 0.2])

    metric_val: Metric = warpshell_comoving(grid_size, world_center, m, big_r_1, big_r_2, r_buff, sigma, smooth_factor, v_warp, do_warp,
                                            grid_scaling)

    assert metric_val.tensor is not None, "Tensor is None."
    assert metric_val.type == "metric", "Type is wrong."
    assert metric_val.coords == "cartesian", "Coordinate system is wrong."
    assert metric_val.index == "covariant", "Index is wrong."

    assert arr_hash(metric_val.tensor) == warpshell_comoving_tensor_hash, "Tensor hash mismatch."
