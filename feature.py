from numbers import Number

import pandas as pd
from pandas import Series, DataFrame


def concat_series(df: DataFrame, *args: Series) -> DataFrame:
    rows = df.shape[0]
    for a in args:
        if a.size != rows:
            raise ValueError(f'All series must have the same length as dataframe. Got: {a.size} ({a.name})')
    return pd.concat([df, pd.concat(args, axis=1, ignore_index=True)], axis=1, ignore_index=True)


def create_polynom(series: Series, degree: int, drop_first=True) -> list[Series]:
    if not issubclass(series.dtype.type, Number):
        raise ValueError('Series must be numeric')
    if degree < 2:
        raise ValueError(f'Degree must be >= 2. Got: {degree}')

    term_series = [series]
    for d in range(2, degree + 1):
        term = term_series[-1] * series
        term.name = f'{series.name}_pow{d}'
        term_series.append(term)
    return term_series[1:] if drop_first else term_series


# Category to mean target mapper
class CMVMapper:
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


# Category to dummies mapper
class CDMapper:
    def __init__(
            self,
            categories: Series
    ):
        self._categories = categories.unique()
        self._columns = [f'{categories.name}_{v}' for v in self._categories]
        self._column_dict = {k: v for k, v in zip(self._categories, self._columns)}

    def map(self, categories: Series, drop_first=True) -> list[Series]:
        dummy_series = [Series(0, dtype=float, name=c) for c in self._columns]
        for cat, ser in zip(self._categories, dummy_series):
            ser[categories == cat] = 1
        return dummy_series[1:] if drop_first else dummy_series

    @property
    def column_dict(self) -> dict:
        return self._column_dict
