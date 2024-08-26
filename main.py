# This is a file to test newly ported stuff
import numpy as np

from Metrics.metric import Metric
from Metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from Metrics.three_plus_one_builder import three_plus_one_builder
from Metrics.three_plus_one_decomposer import three_plus_one_decomposer

metric_val = Metric

alpha = np.absolute(np.random.rand(1, 10, 10, 10))
beta = np.absolute(np.random.rand(1, 3, 1, 10, 10, 10))
gamma = np.absolute(np.random.rand(3, 3, 1, 10, 10, 10))

metric_val.tensor = three_plus_one_builder(alpha, beta, gamma)

print(three_plus_one_decomposer(metric_val))
