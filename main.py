# This is a file to test newly ported stuff
import time
from time import sleep

import h5py
import mat73
import numpy as np
from numpy import dtype

from Analyzer.momentum_flow_lines import momentum_flow_lines
from Analyzer.utils.trace import trace
from Solver import Energy, energy_density, ricci_tensor, tensor_determinant, take_finite_diff_1dir, take_finite_diff_2dirs, \
    met2den
from Solver.get_energy_tensor import get_energy_tensor
from Solver.utils.tensor_inverse import tensor_inverse
from Tests.utils.arr_hash import arr_hash

#np.set_printoptions(threshold=np.inf)
import scipy as sp

from Metrics import Metric, lentz_comoving, modified_time, modified_time_comoving, schwarzschild, van_den_broeck_comoving, \
    van_den_broeck, alcubierre, alcubierre_comoving
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
metric_val: Metric = alcubierre_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))

e_tensor = met2den(metric_val.tensor)
f = mat73.loadmat('C:\\Users\\Lina\\Documents\\MATLAB\\ricci_tensor.mat')
data = np.array(f['R_munu'], dtype=np.float64)

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

"""energy_t = get_energy_tensor(metric_val)

ngridsteps = 4
max_steps = 10000
size = energy_t.tensor.shape
scale_factor = 1/np.max(np.abs(energy_t.tensor[0, 1]))

x, y, z = np.meshgrid(np.arange(1, size[3]+1, ngridsteps), np.arange(1, size[4]+1, ngridsteps),
                      np.arange(1, size[5]+1, ngridsteps), indexing='xy')

start_points = np.array([x, y, z])

paths = momentum_flow_lines(energy_t, start_points, 0.75, max_steps, scale_factor)"""
