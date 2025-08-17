class InternalProblem:
    """Wraps a user provided function to add functionality"""

    def __init__(self, fun, converter):
        self._user_fun = fun
        self._converter = converter

    def fun(self, x):
        params = self._converter.unflatten(x)
        return self._user_fun(params)
