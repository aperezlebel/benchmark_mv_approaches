"""Implement the Task class."""
import pandas as pd
import numpy as np
import logging

from database import dbs


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


class Task:
    """Gather task metadata and the dataframe on which to run the task."""

    def __init__(self, meta):
        self._df = None
        self.meta = meta

    def _load_df(self):
        if self._df is not None:
            return self._df

        db = dbs[self.meta.db]
        db.load(self.meta.df_name)

        df_name = self.meta.df_name

        if self.meta.rename is None:
            logger.info('transform_be is None, reading from encoded df.')
            df = db.encoded_dataframes[df_name]
            parent = db.encoded_parent[df_name]
        else:
            logger.info('transform_be is set, using transform_encode.')
            df, parent = db.rename_encode(df_name, self.meta.rename)

        logger.info(f'df loaded in Task, encoded shape: {df.shape}')

        if self.meta.transform_df is not None:
            df = self.meta.transform_df(df)
            logger.info(f'df loaded in Task, transformed shape: {df.shape}')

        self._df = df
        self._parent = parent

        self._check()
        self._set_drop()

        return self._df

    def _check(self):
        """Check if drop, drop_contains and keep contains feature of the df."""
        predict = self.meta.predict

        parent = self._parent

        drop = None
        keep = None

        if self.meta.drop is not None:
            drop = [f for f in parent.index if parent[f] in self.meta.drop]

        if self.meta.keep is not None:
            keep = [f for f in parent.index if parent[f] in self.meta.keep]

        drop_contains = self.meta.drop_contains
        cols = self._df.columns

        if predict not in cols:
            raise ValueError('predict must be a column name of df.')

        if (drop is not None or drop_contains is not None) and keep is not None:
            raise ValueError('Cannot both keep and drop.')

        for features in [drop, keep]:
            if features is not None:
                if not isinstance(features, list):
                    raise ValueError('drop or keep must be a list or None.')
                elif not all(f in cols for f in features):
                    raise ValueError('Drop/keep must contains column names.')
                elif predict in features:
                    raise ValueError('predict should not be in drop or keep.')

    def _set_drop(self):
        """Compute the features to drop from drop, drop_contains, keep."""
        predict = self.meta.predict
        drop = self.meta.drop
        drop_contains = self.meta.drop_contains
        keep_contains = self.meta.keep_contains
        keep = self.meta.keep
        cols = self._df.columns

        if keep is not None:
            keep = keep+[predict]
            self._drop = [f for f in cols if f not in keep]  # Set drop
        else:
            drop2 = []
            if drop_contains is not None:
                drop_array = np.logical_or.reduce(
                    np.array([cols.str.contains(p) for p in drop_contains])
                )
                drop_series = pd.Series(drop_array, index=cols)
                drop2 = list(drop_series[drop_series].index)
            if drop is None:
                drop = []

            self._drop = drop+drop2  # Set drop

    @property
    def df(self):
        """Full transformed data frame."""
        self._load_df()
        df = self._df.drop(self._drop, axis=1)
        logger.info(f'df shape after dropping cols: {df.shape}')
        return df

    @property
    def X(self):
        """Features used for prediction."""
        return self.df.drop(self.meta.predict, axis=1)

    @property
    def y(self):
        """Feature to predict."""
        return self._df[self.meta.predict]

    def get_infos(self):
        return self.meta.get_infos()
