from copy import deepcopy

from scipy.optimize import minimize as scipy_minimize

from optimini.converter import Converter
from optimini.internal_problem import InternalProblem


def minimize(fun, params, method, options=None):
    """Minimize a function using a given method"""
    options = {} if options is None else options
    converter = Converter(params)
    problem = InternalProblem(fun, converter)
    x0 = converter.flatten(params)
    raw_res = scipy_minimize(
        fun=problem.fun,
        x0=x0,
        method=method,
        options=options,
    )
    res = deepcopy(raw_res)
    res.x = converter.unflatten(res.x)
    return res
