import pandas as pd
import seaborn as sns


class History:
    """History of parameters and function values."""

    def __init__(self):
        self.value = []
        self.params = []

    def add(self, value, params):
        self.value.append(value)
        self.params.append(params)


def history_plot(results, max_n_evals=50, monotone=True):
    """Plot the criterion values of multiple optimizations."""
    data = pd.DataFrame()
    for name, res in results.items():
        values = res.history.value
        df = pd.DataFrame({"value": values, "n_evals": range(len(values))})
        df["name"] = name
        if monotone:
            df["value"] = df["value"].cummin()
        data = pd.concat([data, df])

    if max_n_evals is not None:
        data = data.query(f"n_evals <= {max_n_evals}")

    return sns.lineplot(data=data, x="n_evals", y="value", hue="name")
