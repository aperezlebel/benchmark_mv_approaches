"""Implement abstract class for the databases."""

from abc import ABC, abstractmethod

from features_type import _load_feature_types


class Database(ABC):

    @abstractmethod
    def __init__(self, name='', acronym=''):
        self.dataframes = dict()
        self.features_types = dict()
        self.name = name
        self.acronym = acronym
        self._load_db()
        self._load_feature_types()

    def __getitem__(self, name):
        """Get data frame giving its name."""
        return self.dataframes[name]

    def df_names(self):
        """Get data frames' names."""
        return list(self.dataframes.keys())

    @abstractmethod
    def _load_db(self):
        pass

    @abstractmethod
    def heuristic(self, series):
        """Implement the heuristic for detecting missing values.

        Parameters
        ----------
        series : pandas.Series
            One column of the NHIS dataframe, stored as a pandas.Series object.

        Returns
        -------
        pandas.Series
            A series with same name and index as input series but having values
            in [0, 1, 2] encoding respectively: Not a missing value,
            Not applicable, Not available.

        """
        pass

    def _load_feature_types(self):
        for name in self.df_names():
            try:
                self.features_types[name] = _load_feature_types(self, name)
            except FileNotFoundError:
                print(f'{name}: features types not found. Ignored.')
