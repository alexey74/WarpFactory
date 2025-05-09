import numpy as np
from warpfactory.metrics import AlcubierreMetric
from warpfactory.visualizer import TensorPlotter

# Create spatial grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
z = np.zeros_like(x)
t = 0.0
X, Y = np.meshgrid(x, y)

# Initialize metric
metric = AlcubierreMetric()

# Calculate metric components
components = metric.calculate(
    X, Y, z, t, v_s=2.0, R=1.0, sigma=0.5  # Ship velocity (in c)  # Bubble radius  # Thickness parameter
)

# Visualize metric
plotter = TensorPlotter()
fig = plotter.plot_component(components, "g_tt", x, y)
fig.savefig("metric_tt.png")
