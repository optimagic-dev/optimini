import numpy as np

from optimini.minimize import minimize


def dict_fun(params):
    return params["a"] ** 2 + params["b"] ** 2


def array_fun(params):
    return params @ params


def test_simple_minimize_with_dict_params():
    params = {"a": 1, "b": 2}
    res = minimize(dict_fun, params, method="L-BFGS-B")
    assert isinstance(res.x, dict)
    assert np.allclose(res.x["a"], 0)
    assert np.allclose(res.x["b"], 0)


def test_simple_minimize_with_array_params():
    params = np.array([1, 2])
    res = minimize(array_fun, params, method="L-BFGS-B")
    assert isinstance(res.x, np.ndarray)
    assert np.allclose(res.x, np.array([0, 0]))
