import numpy as np
import pytest

from optimini.history import History, history_plot
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


def test_history_collection():
    params = {"a": 1, "b": 2}
    res = minimize(dict_fun, params, method="L-BFGS-B")
    assert isinstance(res.history, History)
    history_plot({"test": res})


@pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead"])
def test_with_bounds(method):
    params = {"a": 1, "b": 2}
    lb = {"a": -1, "b": 1}
    res = minimize(dict_fun, params, method=method, lower_bounds=lb)
    assert np.allclose(res.x["a"], 0, atol=1e-4)
    assert np.allclose(res.x["b"], 1, atol=1e-4)


def test_unsupported_bounds():
    params = {"a": 1, "b": 2}
    lb = {"a": -1, "b": 1}
    with pytest.raises(ValueError):
        minimize(dict_fun, params, method="CG", lower_bounds=lb)
