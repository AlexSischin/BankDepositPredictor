import numpy as np
from matplotlib import pyplot as plt
from pandas import Series, DataFrame


def plot_density(series: Series) -> None:
    print(series.min())
    series.value_counts(sort=False).sort_index().plot.area()


def plot_correlation(corr: DataFrame):
    corr.style.background_gradient(cmap='coolwarm').format(precision=2)
    fig, ax = plt.subplots()
    ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)), corr.columns)
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
