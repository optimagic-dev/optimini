class InternalProblem:
    """Wraps a user provided function to add functionality"""

    def __init__(self, fun, lb, ub, converter, history):
        self._user_fun = fun
        self._converter = converter
        self._history = history
        self.lower_bounds = converter.flatten(lb)
        self.upper_bounds = converter.flatten(ub)

    def fun(self, x):
        params = self._converter.unflatten(x)
        value = self._user_fun(params)
        self._history.add(value, params)
        return value
