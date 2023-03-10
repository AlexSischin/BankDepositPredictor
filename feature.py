from numbers import Number
from typing import Iterable

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def append_series(
        df: DataFrame,
        series: Series | Iterable[Series]
) -> DataFrame:
    if isinstance(series, Series):
        series = [series]
    return pd.concat([df] + series, axis=1)


def create_polynomial(
        series: Series,
        degree: int,
        drop_first=False,
        dtype=np.float64
) -> list[Series]:
    if not issubclass(series.dtype.type, Number):
        raise ValueError('Series must be numeric')
    if degree < 1:
        raise ValueError(f'Degree must be >= 1. Got: {degree}')

    if dtype is not None:
        series = series.astype(dtype)

    term_series = [series]
    for d in range(2, degree + 1):
        term = term_series[-1] * series
        term.name = f'{series.name}_pow{d}'
        term_series.append(term)
    return term_series[1:] if drop_first else term_series


class MeanTargetMapper:
    def __init__(self,
                 categories: Series,
                 targets: Series,
                 dtype=np.float64
                 ):
        if categories.shape != targets.shape:
            raise ValueError('Categories and target values must have the same shape')
        if not issubclass(targets.dtype.type, Number):
            raise ValueError('Targets must be numeric')

        if dtype is not None:
            targets = targets.astype(dtype)

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
    def __init__(self,
                 categories: Series,
                 drop_first=True,
                 dtype=np.float64
                 ):
        self._categories = categories.unique()
        self._columns = [f'{categories.name}_is_{v}' for v in self._categories]
        self._column_dict = {k: v for k, v in zip(self._categories, self._columns)}
        self._drop_first = drop_first
        self._dtype = dtype

    def map(self, categories: Series) -> list[Series]:
        dummy_series = []
        for cat, col in zip(self._categories, self._columns):
            ser = (categories == cat).astype(np.float64)
            ser.name = col
            dummy_series.append(ser)
        return dummy_series[1:] if self._drop_first else dummy_series

    @property
    def column_dict(self) -> dict:
        return self._column_dict


class SpecialValueMapper:
    def __init__(self,
                 series: Series,
                 value: object,
                 dtype=np.float64
                 ):
        self._value = value
        self._name = f'{series.name}_is_{value}'
        self._dtype = dtype

    def map(self, value_series: Series) -> Series:
        special_value_series = (value_series == self._value).astype(self._dtype)
        special_value_series.name = self._name
        return special_value_series

    @property
    def value(self) -> object:
        return self._value
