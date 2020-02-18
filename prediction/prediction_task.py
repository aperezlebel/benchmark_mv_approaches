"""Implement the prediction task class."""

import pandas as pd
import numpy as np

from database.base import Database

class PredictionTask():

    def __init__(self, db, df_name, to_predict, transform=None, to_drop=None,
                 drop_contains=None, to_keep=None):
        if not isinstance(db, Database):
            raise ValueError('df must be a Database.')

        self._db = db
        self._df_name = df_name
        self._to_predict = to_predict
        self._to_drop = to_drop
        self._drop_contains = drop_contains
        self._to_keep = to_keep

        self._transform = transform
        if transform is None:
            self._transform = lambda x: x

        self._df = None

    def _delayed_load(self):
        if self._df is not None:  # Load already performed
            return  # Ignore

        df_name = self._df_name
        db = self._db

        if not db.is_loaded(df_name):  # If first time db is asked this df
            db.load(df_name)

        self._df = self._transform(db.encoded_dataframes[df_name])

        to_predict = self._to_predict
        to_drop = self._to_drop
        drop_contains = self._drop_contains
        to_keep = self._to_keep

        # Peforms checks on features to drop/keep
        cols = self._df.columns
        if to_predict not in cols:
            raise ValueError('to_predict must be a column name of df.')

        if (to_drop is not None or drop_contains is not None) and to_keep is not None:
            raise ValueError('Cannot both keep and drop.')

        print(cols)
        for features in [to_drop, to_keep]:
            print(features)
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
            # Update to_drop
            self._to_drop = [f for f in cols if f not in to_keep]
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

            # Update to_drop
            self._to_drop = to_drop+to_drop2

    @property
    def df(self):
        self._delayed_load()
        return self._df.drop(self._to_drop, axis=1)

    @property
    def df_plain(self):
        self._delayed_load()
        return self._df

    @property
    def X(self):
        self._delayed_load()
        return self.df.drop(self._to_predict, axis=1)

    @property
    def y(self):
        self._delayed_load()
        return self._df[self._to_predict]
