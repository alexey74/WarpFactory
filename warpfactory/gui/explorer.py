"""Main window for metric exploration."""

import importlib
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
)
from PyQt6.QtCore import Qt

from warpfactory.torch.energy import TorchEnergyTensor

from .plotter import MetricPlotter
from .parameters import ParameterPanel
from .energy import EnergyConditionViewer


class MetricExplorer(QMainWindow):
    """Main window for exploring warp drive metrics."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("WarpFactory Metric Explorer")
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Metric selector
        metric_label = QLabel("Select Metric:")
        self.metric_selector = QComboBox()
        self.metric_selector.setObjectName("metric_selector")
        self.metric_selector.addItems(["Alcubierre", "Lentz", "Van Den Broeck", "Warp Shell", "Minkowski"])
        left_layout.addWidget(metric_label)
        left_layout.addWidget(self.metric_selector)

        # Parameter panel
        self.parameter_panel = ParameterPanel()
        self.parameter_panel.setObjectName("parameter_panel")
        left_layout.addWidget(self.parameter_panel)

        # Add left panel to main layout
        layout.addWidget(left_panel)

        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Metric plotter
        self.plotter = MetricPlotter()
        right_layout.addWidget(self.plotter)

        # Energy condition viewer
        self.energy_viewer = EnergyConditionViewer()
        right_layout.addWidget(self.energy_viewer)

        # Add right panel to main layout
        layout.addWidget(right_panel)

        btn = QPushButton("Update")
        btn.clicked.connect(self.update)
        layout.addWidget(btn)

        # Set up connections
        self.metric_selector.currentTextChanged.connect(self.on_metric_changed)
        self.on_metric_changed("Alcubierre")

    def on_metric_changed(self, metric_name: str):
        """Handle metric selection changes."""
        # Update parameter panel based on metric type
        if metric_name == "Alcubierre":
            params = {"v_s": 2.0, "R": 1.0, "sigma": 0.5}
        elif metric_name == "Lentz":
            params = {"v_s": 2.0, "R": 1.0, "sigma": 0.5}
        elif metric_name == "Van Den Broeck":
            params = {"v_s": 2.0, "R": 1.0, "B": 2.0, "sigma": 0.5}
        elif metric_name == "Warp Shell":
            params = {"v_s": 2.0, "R": 1.0, "thickness": 0.2, "sigma": 0.5}
        else:  # Minkowski
            params = {}

        self.parameter_panel.set_parameters(params)
        self.metric_name = metric_name

    def update(self):
        x = np.linspace(-5, 5, 100)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        t = 0.0

        metric_module = importlib.import_module("warpfactory.metrics.%s" % self.metric_name.lower().replace(" ", "_"))
        metric_class = getattr(metric_module, self.metric_name.replace(" ", "") + "Metric")
        metric = metric_class().calculate(x, y, z, t, **self.parameter_panel.parameters)

        # self.plotter.set_metric(metric)

        energy_tensor = TorchEnergyTensor()
        self.energy_viewer.set_tensor(energy_tensor.calculate(metric, x, y, z))
