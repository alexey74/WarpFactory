# This is a file to test newly ported stuff
import time
from time import sleep

import h5py
import mat73
import numpy as np
from numpy import dtype

from Analyzer.utils.trace import trace
from Solver import Energy, energy_density, ricci_tensor, tensor_determinant, take_finite_diff_1dir, take_finite_diff_2dirs
from Solver.get_energy_tensor import get_energy_tensor
from Solver.utils.tensor_inverse import tensor_inverse
from Tests.utils.arr_hash import arr_hash

#np.set_printoptions(threshold=np.inf)
import scipy as sp

from Metrics import Metric, lentz_comoving, modified_time, modified_time_comoving, schwarzschild, van_den_broeck_comoving, \
    van_den_broeck, alcubierre
from Metrics.warpshell_comoving import warpshell_comoving

# metric_val: Metric = warpshell_comoving(np.array([1, 300, 300, 5]), np.array([1.000692285594456e-11, 30.1, 30.1, 0.6]), np.float64(20/(2*sp.constants.G)*sp.constants.c**2*1/3), np.float64(10), np.float64(20), smooth_factor=np.float64(4000), v_warp=np.float64(0.02), do_warp=True, grid_scaling=np.array([3.335640951981520e-12, 0.2, 0.2, 0.2]))
# print(metric_val)
# metric_val: Metric = van_den_broeck_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
# print(metric_val)
# metric_val: Metric = van_den_broeck(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
# print(metric_val)
# metric_val: Metric = schwarzschild(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.01))
# print(metric_val)
# metric_val: Metric = modified_time_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5), np.float64(0.05))
# print(metric_val)
# metric_val: Metric = modified_time(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5), np.float64(0.05))
# print(metric_val)
# metric_val: Metric = minkowski(np.array([1, 10, 10, 10]), np.array([1, 1, 1, 1]))
# print(metric_val)
# metric_val: Metric = lentz(np.array([5, 30, 30, 2]), np.array([3, 15.5, 15.5, 1.5]), np.float64(0.1))
# print(metric_val)
# metric_val: Metric = lentz_comoving(np.array([1, 30, 30, 2]), np.array([1, 15.5, 15.5, 1.5]), np.float64(0.1))
# print(metric_val)
# metric_val: Metric = alcubierre_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))
# print(metric_val)
# metric_val: Metric = alcubierre(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))
# print(metric_val)
metric_val: Metric = van_den_broeck(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
f_t = mat73.loadmat('array.mat')
data_t = np.array(f_t['n_tensor'], dtype=np.float64)
#e_tensor: np.ndarray[np.float64] = get_energy_tensor(metric_val).tensor

#e_tensor: np.ndarray[np.float64] = ricci_tensor(data_t, tensor_inverse(data_t), np.array([1.0, 1.0, 1.0, 1.0]))
e_tensor: np.ndarray[np.float64] = take_finite_diff_2dirs(data_t[3, 3], 0, 1, np.array([1.0, 1.0, 1.0, 1.0]), False)

f = mat73.loadmat('C:\\Users\\Lina\\Documents\\MATLAB\\array.mat')
data = np.array(f['n_tensor'], dtype=np.float64)

#print((-(e_tensor[:, :, :, 4:] - e_tensor[:, :, :, :-4]) + 8 * (e_tensor[:, :, :, 3:-1] - e_tensor[:, :, :, 1:-3])))
#print((-(data[:, :, :, 4:] - data[:, :, :, :-4]) + 8 * (data[:, :, :, 3:-1] - data[:, :, :, 1:-3])))

# Find indices where values differ
diff_indices = np.where(data != e_tensor)

# Extract differing values
diff_values_a = e_tensor[diff_indices]
diff_values_b = data[diff_indices]

print("Indices with differences:", diff_indices)
print("Differing values in 'a':", diff_values_a, len(diff_values_a), "of", e_tensor.size)
print("Differing values in 'b':", diff_values_b, len(diff_values_b), "of", data.size)
if len(diff_values_a) != 0:
    random_select = np.random.randint(len(diff_values_a), size=25)
    print(diff_values_a[random_select])
    print(diff_values_b[random_select])
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

# arr = np.random.choice(np.arange(0, 100000.0, dtype=np.float64), size=(4, 4, 2, 5, 5, 10), replace=False)
# arr = np.random.choice(np.arange(0, 100000.0, dtype=np.float64), size=(4, 4, 2, 2, 2, 2), replace=False)

# print(c_det(arr))
# print(np.linalg.det(arr))

# print(arr)