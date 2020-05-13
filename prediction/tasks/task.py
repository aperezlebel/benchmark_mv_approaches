"""Implement the Task class."""
import pandas as pd
import numpy as np
import logging

from missing_values import get_missing_values
from database import dbs
from database.base import Database


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


class Task:
    """Gather task metadata and the dataframe on which to run the task."""

    def __init__(self, meta):
        self._df = None
        self.meta = meta
        self._drop = None

    def _load_df(self):
        if self._df is not None:
            return self._df

        db = dbs[self.meta.db]
        db.load(self.meta)

        tag = self.meta.tag

        if self.meta.rename is None:
            logger.info('rename is None, reading from encoded df.')
            df = db.encoded_dataframes[tag]
            parent = db.encoded_parent[tag]
        else:
            logger.info('rename is set, using transform_encode.')
            df, parent = db.rename_encode(tag, self.meta.rename)

        logger.info(f'df loaded in Task, encoded shape: {df.shape}')

        if self.meta.transform_df is not None:
            df = self.meta.transform_df(df)
            logger.info(f'df loaded in Task, transformed shape: {df.shape}')

        # Features addded by the transform function
        extra_features = set(df.columns) - set(parent.index)

        _, to_drop = Database.get_drop_and_keep(
            list(parent.values)+list(extra_features),  # the universe is the parent features
            predict=self.meta.predict,
            keep=self.meta.keep_after_transform,
            drop=self.meta.drop_after_transform,
        )

        # Convert to_drop from parent space to child space
        to_drop = ({i for i, v in parent.items() if v in to_drop}.union(
                   (extra_features).intersection(to_drop)))

        df = df.drop(to_drop, axis=1)
        logger.info(f'df loaded in Task, after drop shape: {df.shape}')

        mv = get_missing_values(df, db.heuristic)
        types = db.feature_types[tag]
        order = db.ordinal_orders.get(tag, None)
        df, _, _, _ = db._encode_df(df, mv, types, order=order, encode='all')

        self._df = df

        self._parent = parent

        return self._df

    @property
    def df(self):
        """Full transformed data frame."""
        return self._load_df()

    @property
    def X(self):
        """Features used for prediction."""
        return self.df.drop(self.meta.predict, axis=1)

    @property
    def y(self):
        """Feature to predict."""
        return self._df[self.meta.predict]

    def get_infos(self):
        infos = self.meta.get_infos()
        infos['df_shape'] = None if self._df is None else repr(self._df.shape)
        return infos

    def is_classif(self):
        return self.meta.classif
