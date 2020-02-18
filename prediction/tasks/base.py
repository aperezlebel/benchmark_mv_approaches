"""Base classes for the task management."""
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import List, Callable


@dataclass(frozen=True)
class TaskMeta():
    """Store the tasks metadata (e.g feature to predict, to drop...)."""

    df_name: str
    predict: str
    drop: List[str] = None
    drop_contains: List[str] = None
    keep: List[str] = None
    transform: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x


@dataclass
class Task:
    """Gather task metadata and the dataframe on which to run the task."""

    _df_untransformed: pd.DataFrame
    meta: TaskMeta

    def __post_init__(self):
        """Transform given df, run checks and set drop according to meta."""
        self._df = self.meta.transform(self._df_untransformed, meta=self.meta)
        self._check()
        self._set_drop()

    def _check(self):
        """Check if drop, drop_contains and keep contains feature of the df."""
        predict = self.meta.predict
        drop = self.meta.drop
        drop_contains = self.meta.drop_contains
        keep = self.meta.keep
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
                    print(features)
                    raise ValueError('Drop/keep must contains column names.')
                elif predict in features:
                    raise ValueError('predict should not be in drop or keep.')

    def _set_drop(self):
        """Compute the features to drop from drop, drop_contains, keep."""
        predict = self.meta.predict
        drop = self.meta.drop
        drop_contains = self.meta.drop_contains
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
        return self._df.drop(self._drop, axis=1)

    @property
    def X(self):
        """Features used for prediction."""
        return self.df.drop(self.meta.predict, axis=1)

    @property
    def y(self):
        """Feature to predict."""
        return self._df[self.meta.predict]
