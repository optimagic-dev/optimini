from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optimini.history import History
from optimini.internal_problem import InternalProblem


@dataclass
class OptimizeResult:
    """An oversimplified optimization result."""

    x: dict | NDArray[np.float64]
    history: History
    fun: float


@dataclass(frozen=True)
class InternalResult:
    x: NDArray[np.float64]
    fun: float


class Algorithm(ABC):
    @abstractmethod
    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        pass
