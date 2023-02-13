import numpy as np
import pandas as pd
from pandas import DataFrame

from feature import OneHotMapper, append_series, create_polynomial, SpecialValueMapper

pd.options.display.max_columns = 100
pd.options.mode.use_inf_as_na = True

COL_AGE = 'age'
COL_JOB = 'job'
COL_MARITAL = 'marital'
COL_EDUCATION = 'education'
COL_DEFAULT = 'default'
COL_BALANCE = 'balance'
COL_HOUSING = 'housing'
COL_LOAN = 'loan'
COL_CONTACT = 'contact'
COL_DAY = 'day'
COL_MONTH = 'month'
COL_DURATION = 'duration'
COL_CAMPAIGN = 'campaign'
COL_PDAYS = 'pdays'
COL_PREVIOUS = 'previous'
COL_POUTCOME = 'poutcome'
COL_Y = 'y'


class MarketingFeatureBuilder:
    def __init__(self, marketing_df: DataFrame):
        self._job_mapper = OneHotMapper(marketing_df[COL_JOB])
        self._marital_mapper = OneHotMapper(marketing_df[COL_MARITAL])
        self._education_mapper = OneHotMapper(marketing_df[COL_EDUCATION])
        self._default_mapper = OneHotMapper(marketing_df[COL_DEFAULT])
        self._housing_mapper = OneHotMapper(marketing_df[COL_HOUSING])
        self._loan_mapper = OneHotMapper(marketing_df[COL_LOAN])
        self._contact_mapper = OneHotMapper(marketing_df[COL_CONTACT])
        self._day_mapper = OneHotMapper(marketing_df[COL_DAY])
        self._month_mapper = OneHotMapper(marketing_df[COL_MONTH])
        self._pdays_mapper = SpecialValueMapper(marketing_df[COL_PDAYS], -1)
        self._poutcome_mapper = OneHotMapper(marketing_df[COL_POUTCOME])

    def build(self, marketing_df: DataFrame) -> DataFrame:
        df = DataFrame()
        a = append_series
        df = a(df, create_polynomial(marketing_df[COL_AGE], 6))  # Age (polynomial)
        df = a(df, self._job_mapper.map(marketing_df[COL_JOB]))  # Job (one-hot)
        df = a(df, self._marital_mapper.map(marketing_df[COL_MARITAL]))  # Marital (one-hot)
        df = a(df, self._education_mapper.map(marketing_df[COL_EDUCATION]))  # Education (one-hot)
        df = a(df, self._default_mapper.map(marketing_df[COL_DEFAULT]))  # Default (one-hot)
        df = a(df, create_polynomial(marketing_df[COL_BALANCE], 5))  # Balance (polynomial)
        df = a(df, self._housing_mapper.map(marketing_df[COL_HOUSING]))  # Housing (one-hot)
        df = a(df, self._loan_mapper.map(marketing_df[COL_LOAN]))  # Loan (one-hot)
        df = a(df, self._contact_mapper.map(marketing_df[COL_CONTACT]))  # Contact (one-hot)
        df = a(df, create_polynomial(marketing_df[COL_DAY], 6))  # Day (polynomial)
        df = a(df, self._day_mapper.map(marketing_df[COL_DAY]))  # Day (one-hot)
        df = a(df, self._month_mapper.map(marketing_df[COL_MONTH]))  # Month (one-hot)
        df = a(df, create_polynomial(marketing_df[COL_DURATION], 5))  # Duration (polynomial)
        df = a(df, create_polynomial(marketing_df[COL_CAMPAIGN], 5))  # Campaign (polynomial)
        df = a(df, create_polynomial(marketing_df[COL_PDAYS], 5))  # Pdays (polynomial)
        df = a(df, self._pdays_mapper.map(marketing_df[COL_PDAYS]))  # Pdays (dummy)
        df = a(df, create_polynomial(marketing_df[COL_PREVIOUS], 5))  # Previous (polynomial)
        df = a(df, self._poutcome_mapper.map(marketing_df[COL_POUTCOME]))  # Poutcome (one-hot)
        return df.astype(np.float64)


def read_deposit_data(file: str):
    marketing_df = pd.read_csv(file, sep=';')
    fb = MarketingFeatureBuilder(marketing_df)
    features_df = fb.build(marketing_df)

    print(features_df)
