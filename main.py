import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

from data import MarketingFeatureBuilder, COL_Y
from learn import LogisticRegressor
from util import cost_graph


def read_deposit_data(file: str, test_examples: int) -> tuple[DataFrame, DataFrame]:
    marketing_df = pd.read_csv(file, sep=';')
    marketing_df.sample(frac=1)
    marketing_df = marketing_df.reset_index(drop=True)

    total = marketing_df.size

    training_set = marketing_df.head(total - test_examples)
    test_set = marketing_df.tail(test_examples)

    return training_set, test_set


def train_model(x_df: DataFrame, y_df: Series) -> LogisticRegressor:
    if x_df.isnull().values.any():
        raise ValueError('X must not contain NaN values')
    if y_df.isnull().values.any():
        raise ValueError('Y must not contain NaN values')

    model = LogisticRegressor()
    x = x_df.to_numpy()
    y = y_df.to_numpy()
    it = 500
    a = 2
    l_ = 0
    hist = model.fit(x, y, it, a, l_)

    costs = np.array([h.cost for h in hist])
    cost_graph(costs)
    plt.show()

    return model


def test_model(model: LogisticRegressor, x_df: DataFrame, y_df: Series):
    x = x_df.to_numpy()
    y = y_df.to_numpy()
    y_hat = model.predict(x)

    relative_error = np.abs(y - y_hat)
    mean_error = np.mean(relative_error)
    standard_error = np.std(relative_error) / np.sqrt(y.size)

    print(f'Mean relative error: {mean_error:.2f} +- {standard_error:.2f}')


def _main():
    train_deposit_df, test_deposit_df = read_deposit_data('dataset/bank-full.csv', 1000)

    feature_builder = MarketingFeatureBuilder(train_deposit_df)

    train_x_df = feature_builder.build(train_deposit_df)
    train_y_series = (train_deposit_df[COL_Y] == 'yes').astype(int)

    test_x_df = feature_builder.build(test_deposit_df)
    test_y_series = (test_deposit_df[COL_Y] == 'yes').astype(int)

    model = train_model(train_x_df, train_y_series)
    test_model(model, test_x_df, test_y_series)


if __name__ == '__main__':
    _main()
