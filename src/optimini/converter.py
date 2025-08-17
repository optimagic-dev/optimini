import numpy as np


class Converter:
    """Class to convert between parameter dictionaries and numpy arrays"""

    def __init__(self, params):
        self.original = params

    def flatten(self, params):
        if isinstance(params, dict):
            params = np.array(list(params.values()))
        return params

    def unflatten(self, x):
        if isinstance(self.original, dict):
            x = dict(zip(self.original.keys(), x.tolist(), strict=False))
        return x
