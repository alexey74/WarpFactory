# This is a file to test newly ported stuff
import time

import numpy as np
import scipy as sp

from Metrics import Metric, sph2cart_diag, alphanumeric_solver
from Metrics.utils.alphanumeric_solver_py import alphanumeric_solver_py
from Metrics.warpshell_comoving import warpshell_comoving
from Solver import legendre_radial_interp

# metric_val: Metric = warpshell_comoving(np.array([1, 300, 300, 5]), np.array([1.000692285594456e-11, 30.1, 30.1, 0.6]), np.float64(20/(2*sp.constants.G)*sp.constants.c**2*1/3), np.float64(10), np.float64(20), smooth_factor=np.float64(4000), v_warp=np.float64(0.02), do_warp=True, grid_scaling=np.array([3.335640951981520e-12, 0.2, 0.2, 0.2]))
# print(metric_val)
# metric_val: Metric = metric_get_van_den_broeck_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
# print(metric_val)
# metric_val: Metric = metric_get_van_den_broeck(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
# print(metric_val)
# metric_val: Metric = metric_get_schwarzschild(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.01))
# print(metric_val)
# metric_val: Metric = metric_get_modified_time_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5), np.float64(0.05))
# print(metric_val)
# metric_val: Metric = metric_get_modified_time(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5), np.float64(0.05))
# print(metric_val)
# metric_val: Metric = metric_get_minkowski(np.array([1, 10, 10, 10]), np.array([1, 1, 1, 1]))
# print(metric_val)
# metric_val: Metric = metric_get_lentz(np.array([5, 30, 30, 2]), np.array([3, 15.5, 15.5, 1.5]), np.float64(0.1))
# print(metric_val)
# metric_val: Metric = metric_get_lentz_comoving(np.array([1, 30, 30, 2]), np.array([1, 15.5, 15.5, 1.5]), np.float64(0.1))
# print(metric_val)
# metric_val: Metric = metric_get_alcubierre_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))
# print(metric_val)
# metric_val: Metric = metric_get_alcubierre(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))
# print(metric_val)

#st = time.process_time_ns()

#for i in range(10000):
    #alphanumeric_solver(big_m, big_p, r)

#et = time.process_time_ns()

#print('Cython', (et - st) / 10 ** 9)

#st = time.process_time_ns()

#for i in range(10000):
    #alphanumeric_solver_py(big_m, big_p, r)

#et = time.process_time_ns()

#print('Python', (et - st) / 10 ** 9)
