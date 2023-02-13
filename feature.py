from numbers import Number
from typing import Iterable

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def append_series(df: DataFrame, series: Series | Iterable[Series]) -> DataFrame:
    if isinstance(series, Series):
        series = [series]
    return pd.concat([df] + series, axis=1)


def create_polynomial(series: Series, degree: int, drop_first=False, dtype=np.float64) -> list[Series]:
    if not issubclass(series.dtype.type, Number):
        raise ValueError('Series must be numeric')
    if degree < 2:
        raise ValueError(f'Degree must be >= 2. Got: {degree}')

    if dtype is not None:
        series = series.astype(dtype)

    term_series = [series]
    for d in range(2, degree + 1):
        term = term_series[-1] * series
        term.name = f'{series.name}_pow{d}'
        term_series.append(term)
    return term_series[1:] if drop_first else term_series


class MeanTargetMapper:
    def __init__(
            self,
            categories: Series,
            targets: Series
    ):
        if categories.shape != targets.shape:
            raise ValueError('Categories and target values must have the same shape')
        if not issubclass(targets.dtype.type, Number):
            raise ValueError('Targets must be numeric')

        c_name, v_name = categories.name, targets.name
        df = pd.concat((categories, targets), axis=1)
        df = df.groupby(c_name).mean(numeric_only=True)

        self._name = f'{c_name}_mean_{v_name}'
        self._value_dict = df.to_dict()[v_name]

    def map(self, categories: Series) -> Series:
        mapped_categories = categories.map(self._value_dict)
        mapped_categories.name = self._name
        return mapped_categories

    @property
    def value_dict(self) -> dict:
        return self._value_dict


class OneHotMapper:
    def __init__(
            self,
            categories: Series,
            drop_first=True
    ):
        self._categories = categories.unique()
        self._columns = [f'{categories.name}_is_{v}' for v in self._categories]
        self._column_dict = {k: v for k, v in zip(self._categories, self._columns)}
        self._drop_first = drop_first

    def map(self, categories: Series) -> list[Series]:
        zeros = np.zeros(shape=categories.shape)
        dummy_series = [Series(zeros, dtype=int, name=c) for c in self._columns]
        for cat, ser in zip(self._categories, dummy_series):
            ser[categories == cat] = 1
        return dummy_series[1:] if self._drop_first else dummy_series

    @property
    def column_dict(self) -> dict:
        return self._column_dict


class ZScoreNormalizer:
    def __init__(self, feature: Series):
        self._m = feature.mean()
        self._sd = feature.std()

    def scale(self, feature: Series):
        return (feature - self._m) / self._sd


class SpecialValueMapper:
    def __init__(self, series: Series, value: Number):
        self._value = value
        self._name = f'{series.name}_is_{value}'

    def map(self, value_series: Series) -> Series:
        special_value_series = (value_series == self._value).astype(int)
        special_value_series.name = self._name
        return special_value_series

    @property
    def value(self) -> Number:
        return self._value
