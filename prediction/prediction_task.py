"""Implement the prediction task class."""

import pandas as pd
import numpy as np


class PredictionTask():

    def __init__(self, df, to_predict, to_drop=None, drop_contains=None, to_keep=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pandas DataFrame.')

        self._df = df

        cols = df.columns
        if to_predict not in cols:
            raise ValueError('to_predict must be a column name of df.')
        self.to_predict = to_predict

        if (to_drop is not None or drop_contains is not None) and to_keep is not None:
            raise ValueError('Cannot both keep and drop.')

        for features in [to_drop, to_keep]:
            if features is not None:
                if not isinstance(features, list):
                    raise ValueError('to_drop or to_keep must be a list or None.')
                elif not all(f in cols for f in features):
                    print(features)
                    raise ValueError('Values of to_drop or to_keep must be df columns names.')
                elif to_predict in features:
                    raise ValueError('to_predict should not be in to_drop or to_keep.')

        if to_keep is not None:
            to_keep = to_keep+[to_predict]
            self.to_drop = [f for f in cols if f not in to_keep]
        else:
            to_drop2 = []
            if drop_contains is not None:
                drop_array = np.logical_or.reduce(
                    np.array([cols.str.contains(p) for p in drop_contains])
                )
                drop_series = pd.Series(drop_array, index=cols)
                to_drop2 = list(drop_series[drop_series].index)
            if to_drop is None:
                to_drop = []

            self.to_drop = to_drop+to_drop2

    @property
    def df(self):
        return self._df.drop(self.to_drop, axis=1)

    @property
    def df_plain(self):
        return self._df

    @property
    def X(self):
        return self.df.drop(self.to_predict, axis=1)

    @property
    def y(self):
        return self._df[self.to_predict]
