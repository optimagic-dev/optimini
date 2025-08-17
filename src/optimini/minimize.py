from optimini.algorithms import OPTIMIZER_REGISTRY
from optimini.converter import Converter
from optimini.history import History
from optimini.internal_problem import InternalProblem
from optimini.utils import OptimizeResult


def minimize(fun, params, method, lower_bounds=None, upper_bounds=None, options=None):
    options = {} if options is None else options
    algo = OPTIMIZER_REGISTRY[method](**options)

    _fail_if_incompatible_bounds(lower_bounds, upper_bounds, algo)

    converter = Converter(params)
    history = History()
    problem = InternalProblem(
        fun,
        lower_bounds,
        upper_bounds,
        converter,
        history,
    )
    x0 = converter.flatten(params)
    raw_res = algo._solve_internal_problem(problem, x0)
    res = OptimizeResult(
        x=converter.unflatten(raw_res.x),
        history=history,
        fun=raw_res.fun,
    )
    return res


def _fail_if_incompatible_bounds(lower_bounds, upper_bounds, algo):
    supports_bounds = algo.__algo_info__.supports_bounds
    needs_bounds = algo.__algo_info__.needs_bounds
    if supports_bounds and needs_bounds:
        if lower_bounds is None or upper_bounds is None:
            raise ValueError("Bounds are required for this algorithm")
    if not supports_bounds and (lower_bounds is not None or upper_bounds is not None):
        raise ValueError("Bounds are not supported for this algorithm")
