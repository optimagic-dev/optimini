from dataclasses import dataclass


@dataclass
class AlgoInfo:
    """An oversimplified collection of algorithm properties"""

    name: str
    supports_bounds: bool
    needs_bounds: bool


def minimizer(name, supports_bounds, needs_bounds):
    """Decorator to mark minimizers and add algorithm information"""

    def decorator(cls):
        algo_info = AlgoInfo(name, supports_bounds, needs_bounds)
        cls.__algo_info__ = algo_info
        return cls

    return decorator
