"""Interactive metric visualization widget."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class MetricPlotter(QWidget):
    """Widget for interactive metric visualization."""

    def __init__(self):
        """Initialize the plotter widget."""
        super().__init__()
        self.current_component = "g_tt"
        self.colormap = "viridis"
        self.figure = Figure()

        x = self.x = np.linspace(-1, 1, 10)

        self.components = {
            "g_tt": -np.ones_like(x),
            "g_tx": np.zeros_like(x),
            "g_xx": np.ones_like(x),
            "g_yy": np.ones_like(x),
            "g_zz": np.ones_like(x),
        }

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Component selector
        comp_label = QLabel("Component:")
        self.comp_selector = QComboBox()
        self.comp_selector.addItems(["g_tt", "g_tx", "g_xx", "g_yy", "g_zz"])
        layout.addWidget(comp_label)
        layout.addWidget(self.comp_selector)

        # Matplotlib canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        # Colormap selector
        cmap_label = QLabel("Colormap:")
        self.cmap_selector = QComboBox()
        self.cmap_selector.addItems(["viridis", "plasma"])  # "redblue", "warp",
        layout.addWidget(cmap_label)
        layout.addWidget(self.cmap_selector)

        # Connect signals
        self.comp_selector.currentTextChanged.connect(self.plot_component)
        self.cmap_selector.currentTextChanged.connect(self.set_colormap)

        # self.set_metric(self.components)

    def set_metric(self, components: dict):
        """Set the metric components to display.

        Parameters
        ----------
        components : dict
            Dictionary of metric components
        """
        self.components = components
        self.plot_component(self.current_component)

    def plot_component(self, component: str):
        """Plot a specific metric component.

        Parameters
        ----------
        component : str
            Name of component to plot
        """
        if not hasattr(self, "components"):
            return

        self.current_component = component
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        data = self.components[component]
        im = ax.imshow(data, cmap=self.colormap)
        self.figure.colorbar(im)
        ax.set_title(f"Metric Component {component}")

        self.canvas.draw()

    def set_colormap(self, cmap: str):
        """Set the colormap for visualization.

        Parameters
        ----------
        cmap : str
            Name of colormap to use
        """
        self.colormap = cmap
        if hasattr(self, "components"):
            self.plot_component(self.current_component)
