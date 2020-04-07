"""Implement the TaskMeta class."""
import pandas as pd

from dataclasses import dataclass
from typing import List, Callable, Dict


@dataclass(frozen=True)
class TaskMeta():
    """Store the tasks metadata (e.g feature to predict, to drop...)."""

    name: str  # Name of the task
    db: str
    df_name: str
    predict: str
    drop: List[str] = None
    drop_contains: List[str] = None
    keep_contains: List[str] = None
    keep: List[str] = None
    rename: Dict[str, str] = None  #lambda x: x
    transform: Callable[[pd.DataFrame], pd.DataFrame] = None  #lambda x: x

    def get_infos(self):
        """
        Return a dict of some printable properties.

        Used to next dump task infos in a file.
        """
        props = ['db', 'df_name', 'predict', 'drop', 'drop_contains', 'rename',
                 'keep_contains', 'keep']
        return dict(filter(lambda x: x[0] in props, self.__dict__.items()))

    @property
    def tag(self):
        return f'{self.db}/{self.name}'

    def transform_df(self, df):
        if self.transform is None:
            return None

        return self.transform(df, meta=self)
