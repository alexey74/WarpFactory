import numpy as np

from dataclasses import dataclass

@dataclass
class Energy:
    name: str
    type: str = None
    tensor: np.ndarray[np.float64] = None
    coords: str = None
    index: str = None
    date: str = None
    order: int = None