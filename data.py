import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import positive_class_probability, count_bar

pd.options.display.max_columns = 50
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


def read_deposit_data(file: str):
    deposit_df = pd.read_csv(file, sep=';')
    # print(deposit_df.info())
    # positive_class_probability(deposit_df[COL_PREVIOUS], deposit_df[COL_Y], 'yes', categorical=False, bins=np.arange(0, 20))
    count_bar(deposit_df[COL_PREVIOUS], categorical=False, bins=np.arange(0, 20))
    # plot_correlation(deposit_df.corr(numeric_only=True))
    plt.show()
