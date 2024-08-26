# This is a file to test newly ported stuff
import numpy as np

from Metrics.Schwarzschild.metric_get_schwarzschild import metric_get_schwarzschild

print(metric_get_schwarzschild(np.array([1, 10, 10, 10]), np.array([1, 2, 5, 3]), np.double(2.1)).tensor)
