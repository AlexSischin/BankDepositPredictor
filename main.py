import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

from data import MarketingFeatureBuilder, COL_Y
from learn import LogisticRegressor, calc_confusion_matrix
from util import cost_graph, confusion_matrix

log = logging.getLogger(__name__)


def read_deposit_data(file: str, test_examples: int) -> tuple[DataFrame, DataFrame]:
    marketing_df = pd.read_csv(file, sep=';')
    marketing_df = marketing_df.sample(frac=1)
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

    for h in hist:
        log.info(f'\n{h}')

    costs = np.array([h.cost for h in hist])
    cost_graph(costs)
    plt.show()

    return model


def test_model(model: LogisticRegressor, x_df: DataFrame, y_df: Series, threshold=0.5):
    x = x_df.to_numpy()
    y = y_df.to_numpy()
    y_hat = model.predict(x)
    y_pr = (y_hat > threshold).astype(int)

    conf_matrix = calc_confusion_matrix(y, y_pr)
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / y.size
    sensitivity = conf_matrix[1, 1] / (conf_matrix[:, 1].sum())
    specificity = conf_matrix[0, 0] / (conf_matrix[:, 0].sum())

    print(f'Accuracy: {accuracy * 100:.1f}%')
    print(f'Sensitivity: {sensitivity * 100:.1f}%')
    print(f'Specificity: {specificity * 100:.1f}%')
    confusion_matrix(conf_matrix)
    plt.show()


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
    logfile = "./log/log.txt"
    np.set_printoptions(linewidth=np.inf)
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
    _main()
