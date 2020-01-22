"""Implement abstract class for the databases."""

from abc import ABC, abstractmethod


class Database(ABC):

    @abstractmethod
    def __init__(self):
        self.tables = dict()
        self.name = ''
        self.acronym = ''
        self._load()

    def __getitem__(self, name):
        """Get data frame giving its name."""
        return self.tables[name]

    def tables_names(self):
        """Get tables' names."""
        return list(self.tables.keys())

    @abstractmethod
    def _load(self):
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
