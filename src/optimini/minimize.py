from optimini.algorithms import OPTIMIZER_REGISTRY
from optimini.converter import Converter
from optimini.history import History
from optimini.internal_problem import InternalProblem
from optimini.utils import OptimizeResult


def minimize(fun, params, method, options=None):
    """Minimize a function using a given method"""
    options = {} if options is None else options
    converter = Converter(params)
    history = History()
    problem = InternalProblem(fun, converter, history)
    x0 = converter.flatten(params)
    algo = OPTIMIZER_REGISTRY[method](**options)
    raw_res = algo._solve_internal_problem(problem, x0)
    res = OptimizeResult(
        x=converter.unflatten(raw_res.x),
        history=history,
        fun=raw_res.fun,
    )
    return res
