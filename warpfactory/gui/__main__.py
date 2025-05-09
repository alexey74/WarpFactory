from .explorer import MetricExplorer
import sys
from PyQt6.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    explorer = MetricExplorer()
    explorer.show()
    sys.exit(app.exec())
