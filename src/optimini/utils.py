from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimini.history import History


@dataclass
class OptimizeResult:
    """An oversimplified optimization result."""

    x: dict | NDArray[np.float64]
    history: History
    fun: float
