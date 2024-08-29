# This is a file to test newly ported stuff
import numpy as np

from Metrics.Alcubierre.metric_get_alcubierre import metric_get_alcubierre
from Metrics.three_plus_one_decomposer import three_plus_one_decomposer

print(three_plus_one_decomposer(metric_get_alcubierre(np.array([5, 20, 20, 20]), np.array([3, 10.5, 10.5, 10.5]), np.float64(4.9),
                                                      np.float64(5), np.float64(0.5))))
