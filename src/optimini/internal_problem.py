class InternalProblem:
    """Wraps a user provided function to add functionality"""

    def __init__(self, fun, converter, history):
        self._user_fun = fun
        self._converter = converter
        self._history = history

    def fun(self, x):
        params = self._converter.unflatten(x)
        value = self._user_fun(params)
        self._history.add(value, params)
        return value
