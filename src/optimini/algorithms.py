from dataclasses import dataclass

import nlopt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import minimize as scipy_minimize

from optimini import mark
from optimini.internal_problem import InternalProblem
from optimini.utils import Algorithm, InternalResult


@mark.minimizer(
    name="L-BFGS-B",
    supports_bounds=True,
    needs_bounds=False,
)
@dataclass(frozen=True)
class SciPyLBFGSB(Algorithm):
    convergence_ftol: float = 1e-8
    stopping_maxiter: int = 10_000
    limited_memory_length: int = 12
    # more options here ...

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        bounds = _get_scipy_bounds(problem.lower_bounds, problem.upper_bounds)

        options = {
            "maxcor": self.limited_memory_length,
            "ftol": self.convergence_ftol,
            "maxiter": self.stopping_maxiter,
        }
        res = scipy_minimize(
            fun=problem.fun,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
        return InternalResult(x=res.x, fun=res.fun)


@mark.minimizer(
    name="CG",
    supports_bounds=False,
    needs_bounds=False,
)
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


@mark.minimizer(
    name="Nelder-Mead",
    supports_bounds=True,
    needs_bounds=False,
)
@dataclass(frozen=True)
class SciPyNelderMead(Algorithm):
    stopping_maxiter: int = 10_000
    convergence_ftol: float = 1e-8
    adaptive: bool = True
    # more options here ...

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        bounds = _get_scipy_bounds(problem.lower_bounds, problem.upper_bounds)

        options = {
            "maxiter": self.stopping_maxiter,
            "ftol": self.convergence_ftol,
            "adaptive": self.adaptive,
        }

        res = scipy_minimize(
            fun=problem.fun, x0=x0, method="Nelder-Mead", bounds=bounds, options=options
        )
        return InternalResult(x=res.x, fun=res.fun)


def _get_scipy_bounds(lower_bounds, upper_bounds):
    if lower_bounds is None and upper_bounds is None:
        return None
    if lower_bounds is None:
        lower_bounds = -np.inf
    if upper_bounds is None:
        upper_bounds = np.inf

    return ScipyBounds(lower_bounds, upper_bounds)


@mark.minimizer(
    name="nlopt_bobyqa",
    supports_bounds=True,
    needs_bounds=False,
)
@dataclass(frozen=True)
class NloptBobyqa(Algorithm):
    stopping_maxfun: int = 100
    convergence_ftol_rel: float = 1e-4
    # more options here ...

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[np.float64]
    ) -> InternalResult:
        def func(x, grad):
            if grad.size > 0:
                fun, jac = problem.fun_and_jac(x)
                grad[:] = jac
            else:
                fun = problem.fun(x)
            return fun

        opt = nlopt.opt(nlopt.LN_BOBYQA, x0.shape[0])
        opt.set_min_objective(func)
        if self.convergence_ftol_rel is not None:
            opt.set_ftol_rel(self.convergence_ftol_rel)
        if self.stopping_maxfun is not None:
            opt.set_maxeval(self.stopping_maxfun)
        if problem.lower_bounds is not None:
            opt.set_lower_bounds(problem.lower_bounds)
        if problem.upper_bounds is not None:
            opt.set_upper_bounds(problem.upper_bounds)

        solution_x = opt.optimize(x0)

        return InternalResult(solution_x, opt.last_optimum_value())


OPTIMIZER_REGISTRY = {
    "L-BFGS-B": SciPyLBFGSB,
    "CG": SciPyCG,
    "Nelder-Mead": SciPyNelderMead,
    "bobyqa": NloptBobyqa,
}
