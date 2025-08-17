from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize

from optimini.internal_problem import InternalProblem
from optimini.utils import Algorithm, InternalResult


@dataclass(frozen=True)
class SciPyLBFGSB(Algorithm):
    convergence_ftol: float = 1e-8
    stopping_maxiter: int = 10_000
    limited_memory_length: int = 12
    # more options here ...

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        options = {
            "maxcor": self.limited_memory_length,
            "ftol": self.convergence_ftol,
            "maxiter": self.stopping_maxiter,
        }
        res = scipy_minimize(
            fun=problem.fun,
            x0=x0,
            method="L-BFGS-B",
            options=options,
        )
        return InternalResult(x=res.x, fun=res.fun)


@dataclass(frozen=True)
class SciPyCG(Algorithm):
    convergence_gtol: float = 1e-8
    stopping_maxiter: int = 10_000
    # more options here ...

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        options = {
            "gtol": self.convergence_gtol,
            "maxiter": self.stopping_maxiter,
        }
        res = scipy_minimize(fun=problem.fun, x0=x0, method="CG", options=options)
        return InternalResult(x=res.x, fun=res.fun)


@dataclass(frozen=True)
class SciPyNelderMead(Algorithm):
    stopping_maxiter: int = 10_000
    convergence_ftol: float = 1e-8
    adaptive: bool = True
    # more options here ...

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        options = {
            "maxiter": self.stopping_maxiter,
            "fatol": self.convergence_ftol,
            "adaptive": self.adaptive,
        }

        res = scipy_minimize(
            fun=problem.fun, x0=x0, method="Nelder-Mead", options=options
        )
        return InternalResult(x=res.x, fun=res.fun)


OPTIMIZER_REGISTRY = {
    "L-BFGS-B": SciPyLBFGSB,
    "CG": SciPyCG,
    "Nelder-Mead": SciPyNelderMead,
}
