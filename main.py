# This is a file to test newly ported stuff
import numpy as np
import scipy as sp

from Metrics.Alcubierre.metric_get_alcubierre import metric_get_alcubierre
from Metrics.Alcubierre.metric_get_alcubierre_comoving import metric_get_alcubierre_comoving
from Metrics.Lentz.metric_get_lentz import metric_get_lentz
from Metrics.Lentz.metric_get_lentz_comoving import metric_get_lentz_comoving
from Metrics.Minkowski.metric_get_minkowski import metric_get_minkowski
from Metrics.ModifiedTime.metric_get_modified_time import metric_get_modified_time
from Metrics.ModifiedTime.metric_get_modified_time_comoving import metric_get_modified_time_comoving
from Metrics.Schwarzschild.metric_get_schwarzschild import metric_get_schwarzschild
from Metrics.VanDenBroeck.metric_get_van_den_broeck import metric_get_van_den_broeck
from Metrics.VanDenBroeck.metric_get_van_den_broeck_comoving import metric_get_van_den_broeck_comoving
from Metrics.WarpShell.metric_get_warpshell_comoving import metric_val_get_warpshell_comoving
from Metrics.metric import Metric
from Metrics.three_plus_one_decomposer import three_plus_one_decomposer

# metric_val: Metric = metric_val_get_warpshell_comoving(np.array([1, 300, 300, 5]), np.array([1.000692285594456e-11, 30.1, 30.1, 0.6]), np.float64(20/(2*sp.constants.G)*sp.constants.c**2*1/3), np.float64(10), np.float64(20), smooth_factor=np.float64(4000), v_warp=np.float64(0.02), do_warp=True, grid_scaling=np.array([3.335640951981520e-12, 0.2, 0.2, 0.2]))
# print(metric_val)
metric_val: Metric = metric_get_van_den_broeck_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
print(metric_val)
metric_val: Metric = metric_get_van_den_broeck(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.1), np.float64(2), np.float64(1), np.float64(5), np.float64(1), np.float64(0.5))
print(metric_val)
metric_val: Metric = metric_get_schwarzschild(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.01))
print(metric_val)
metric_val: Metric = metric_get_modified_time_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5), np.float64(0.05))
print(metric_val)
metric_val: Metric = metric_get_modified_time(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5), np.float64(0.05))
print(metric_val)
metric_val: Metric = metric_get_minkowski(np.array([1, 10, 10, 10]), np.array([1, 1, 1, 1]))
print(metric_val)
metric_val: Metric = metric_get_lentz(np.array([5, 30, 30, 2]), np.array([3, 15.5, 15.5, 1.5]), np.float64(0.1))
print(metric_val)
metric_val: Metric = metric_get_lentz_comoving(np.array([1, 30, 30, 2]), np.array([1, 15.5, 15.5, 1.5]), np.float64(0.1))
print(metric_val)
metric_val: Metric = metric_get_alcubierre_comoving(np.array([1, 20, 20, 20]), np.array([1, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))
print(metric_val)
metric_val: Metric = metric_get_alcubierre(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(0.5), np.float64(5), np.float64(0.5))
print(metric_val)