"""Gather all NHIS related functions."""

import pandas as pd
import numpy as np

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING


class NHIS(Database):

    def __init__(self, load=None):
        data_folder = 'NHIS2017/data/'

        paths = {
            'family': data_folder+'family/familyxx.csv',
            'child': data_folder+'child/samchild.csv',
            'adult': data_folder+'adult/samadult.csv',
            'person': data_folder+'person/personsx.csv',
            'household': data_folder+'household/househld.csv',
            'injury': data_folder+'injury/injpoiep.csv',
        }
        sep = ','

        super().__init__(
            name='National Health Interview Survey',
            acronym='NHIS',
            paths=paths,
            sep=sep,
            load=load
            )

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

    def _encode(self):
        super()._encode()
