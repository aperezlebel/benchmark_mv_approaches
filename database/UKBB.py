"""Gather all UKBB related functions."""

import pandas as pd
import numpy as np

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING


class _UKBB(Database):

    def __init__(self):
        super().__init__('UK BioBank', 'UKBB')
        self._encode()

    def _load_db(self, encode=True):
        """Load the UKBB database."""
        data_folder = 'UKBB/ukbb_tabular/csv/'

        self.dataframes = {
            '24440': pd.read_csv(data_folder+'ukb24440.csv', sep=',')
        }

    @staticmethod
    def heuristic(series):
        # The series storing the type of missing values
        series_mv = pd.Series(NOT_MISSING, index=series.index,
                              name=series.name)

        series_mv[series.isna()] = NOT_AVAILABLE

        print(series.name)

        return series_mv


UKBB = _UKBB()
