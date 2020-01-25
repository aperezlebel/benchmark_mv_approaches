"""Gather all NHIS related functions."""

import pandas as pd
import numpy as np

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING


class _NHIS(Database):

    def __init__(self):
        super().__init__('National Health Interview Survey', 'NHIS')

    def _load_db(self):
        """Load the NHIS database."""
        data_folder = 'NHIS2017/data/'

        self.dataframes = {
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
        series_mv = pd.Series(NOT_MISSING, index=series.index,
                              name=series.name)

        # Type 1 missing values
        series_mv[series.isna()] = NOT_APPLICABLE

        # Type 2 missing values
        # This rule is not applicable for mixed types columns
        if series.dtype == np.object:
            return series_mv

        val_max = series.max()

        if val_max < 10:  # Type 1 column
            series_mv[series == 7] = NOT_AVAILABLE
            series_mv[series == 8] = NOT_AVAILABLE
            series_mv[series == 9] = NOT_AVAILABLE

        if val_max < 100:  # Type 2 column
            series_mv[series == 97] = NOT_AVAILABLE
            series_mv[series == 98] = NOT_AVAILABLE
            series_mv[series == 99] = NOT_AVAILABLE

        return series_mv


NHIS = _NHIS()
