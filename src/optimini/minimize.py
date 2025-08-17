from scipy.optimize import minimize as scipy_minimize

from optimini.converter import Converter
from optimini.history import History
from optimini.internal_problem import InternalProblem
from optimini.utils import OptimizeResult


def minimize(fun, params, method, options=None):
    """Minimize a function using a given method"""
    options = {} if options is None else options
    converter = Converter(params)
    history = History()
    internal_fun = InternalProblem(fun, converter, history)
    x0 = converter.flatten(params)
    raw_res = scipy_minimize(
        fun=internal_fun.fun,
        x0=x0,
        method=method,
        options=options,
    )
    res = OptimizeResult(
        x=converter.unflatten(raw_res.x),
        history=history,
        fun=raw_res.fun,
    )
    return res
