"""Gather all NHIS related functions."""

import pandas as pd
import numpy as np

from .base import Database


class _NHIS(Database):

    def __init__(self):
        super().__init__()
        self.name = 'National Health Interview Survey'
        self.acronym = 'NHIS'

    def _load(self):
        """Load the NHIS database."""
        data_folder = 'NHIS2017/data/'

        self.tables = {
            'family': pd.read_csv(data_folder+'family/familyxx.csv'),
            'child': pd.read_csv(data_folder+'child/samchild.csv'),
            'adult': pd.read_csv(data_folder+'adult/samadult.csv'),
            'person': pd.read_csv(data_folder+'person/personsx.csv'),
            'household': pd.read_csv(data_folder+'household/househld.csv'),
            'injury': pd.read_csv(data_folder+'injury/injpoiep.csv'),
        }

    @staticmethod
    def heuristic(series):
        # The series storing the type of missing values
        series_mv = pd.Series(0, index=series.index, name=series.name)

        # Type 1 missing values
        series_mv[series.isna()] = 1  # Type 1 missing values

        # Type 2 missing values
        # This rule is not applicable for mixed types columns
        if series.dtype == np.object:
            return series_mv

        val_max = series.max()

        if val_max < 10:  # Type 1 column
            series_mv[series == 7] = 2
            series_mv[series == 8] = 2
            series_mv[series == 9] = 2

        if val_max < 100:  # Type 2 column
            series_mv[series == 97] = 2
            series_mv[series == 98] = 2
            series_mv[series == 99] = 2

        return series_mv


NHIS = _NHIS()
