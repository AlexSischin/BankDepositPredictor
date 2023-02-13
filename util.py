from collections import defaultdict
from numbers import Number

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series, DataFrame, Interval


def _float_interval_to_str(i: Interval):
    return f'({i.left},{i.right}]'


def _int_interval_to_str(i: Interval):
    return f'({int(np.floor(i.left))},{int(np.floor(i.right))}]'


def positive_class_probability(independent_vals: Series, class_vals: Series, positive_class: object, categorical,
                               bins=10, integer=True, sort=True):
    _, ax = plt.subplots()
    ax.set_xlabel(f'{independent_vals.name}')
    ax.set_ylabel(f'Probability of class: {positive_class}')

    class_map = defaultdict(lambda: 0)
    class_map[positive_class] = 1
    class_vals = class_vals.map(class_map)

    if not categorical:
        if isinstance(bins, Number):
            bins = np.linspace(independent_vals.min(), independent_vals.max(), bins + 1)
        if integer:
            bins = np.rint(bins).astype(np.int64)
        interval_mapper = _int_interval_to_str if integer else _float_interval_to_str
        independent_vals = pd.cut(independent_vals, bins, include_lowest=True).map(interval_mapper)

    col_values, col_class, col_probabilities = 'values', 'class', 'probabilities'

    class_df = pd.concat([independent_vals, class_vals], axis=1)
    class_df.columns = col_values, col_class
    probabilities_df = class_df.groupby(by=col_values).mean().reset_index()
    probabilities_df.columns = col_values, col_probabilities

    if categorical and sort:
        probabilities_df.sort_values(by=col_probabilities, ascending=False, inplace=True)

    x = probabilities_df[col_values]
    y = probabilities_df[col_probabilities]
    x_numeric = np.arange(x.size)
    ax.bar(x_numeric, y)
    ax.set_xticks(x_numeric, x, rotation=45, ha='right')


def count_bar(series: Series, categorical, bins=10, integer=True):
    _, ax = plt.subplots()
    ax.set_xlabel(f'{series.name}')
    ax.set_ylabel(f'Occurrences')

    if not categorical:
        if isinstance(bins, Number):
            bins = np.linspace(series.min(), series.max(), bins + 1)
        if integer:
            bins = np.rint(bins).astype(np.int64)
        interval_mapper = _int_interval_to_str if integer else _float_interval_to_str
        series = pd.cut(series, bins, include_lowest=True).map(interval_mapper)

    count_df = series.value_counts(sort=False).sort_index().reset_index()
    col_values, col_occurrences = 'values', 'occurrences'
    count_df.columns = col_values, col_occurrences

    if categorical:
        count_df.sort_values(by=col_occurrences, ascending=False, inplace=True)

    x = count_df[col_values]
    y = count_df[col_occurrences]
    x_numeric = np.arange(x.size)
    ax.bar(x_numeric, y)
    ax.set_xticks(x_numeric, x, rotation=45, ha='right')


def correlation_matrix(corr: DataFrame):
    corr.style.background_gradient(cmap='coolwarm').format(precision=2)
    fig, ax = plt.subplots()
    ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)), corr.columns)
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for (i, j), z in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))


def cost_graph(costs: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(costs) + 1), costs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
