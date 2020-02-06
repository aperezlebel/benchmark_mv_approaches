"""Implement the prediction task class."""

import pandas as pd


class PredictionTask():

    def __init__(self, df, to_predict, to_drop=None, to_keep=None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pandas DataFrame.')

        self._df = df

        columns = df.columns
        if to_predict not in columns:
            raise ValueError('to_predict must be a column name of df.')
        self.to_predict = to_predict

        if to_drop is not None and to_keep is not None:
            raise ValueError('to_drop and to_keep cannot be both non None.')

        features = to_drop
        if to_keep is not None:
            features = to_keep

        if features is not None:
            if not isinstance(features, list):
                raise ValueError('to_drop or to_keep must be a list or None.')
            elif not all(f in columns for f in features):
                raise ValueError('Values of to_drop or to_keep must be df columns names.')
            elif to_predict in features:
                raise ValueError('to_predict should not be in to_drop or to_keep.')

        self.to_drop = to_drop
        if to_keep is not None:
            to_keep = to_keep+[to_predict]
            self.to_drop = [f for f in columns if f not in to_keep]

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
