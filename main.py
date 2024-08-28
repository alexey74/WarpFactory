# This is a file to test newly ported stuff
import numpy as np

from Metrics.VanDenBroeck.metric_get_van_den_broeck import metric_get_van_den_broeck

print(metric_get_van_den_broeck(np.array([1, 10, 10, 10]), np.array([1, 2, 5, 3]), np.float64(4.9), np.float64(167),
                            np.float64(2.37), np.float64(152),
                            np.float64(5.8), np.float64(0.0087)).tensor)
