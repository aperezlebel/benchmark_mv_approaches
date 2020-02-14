"""Gather all UKBB related functions."""

import pandas as pd
import numpy as np

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING


class UKBB(Database):

    def __init__(self, load=None):
        data_folder = 'UKBB/ukbb_tabular/csv/'
        paths = {
            '24440': data_folder+'ukb24440.csv',
            '25908': data_folder+'ukb25908.csv',
            '28435': data_folder+'ukb28435.csv',
            '32309': data_folder+'ukb32309.csv',
            '35375': data_folder+'ukb35375.csv',
            '38276': data_folder+'ukb38276.csv',
        }
        sep = ','

        super().__init__(
            name='UK BioBank',
            acronym='UKBB',
            paths=paths,
            sep=sep,
            load=load)

    @staticmethod
    def heuristic(series):
        # The series storing the type of missing values
        series_mv = pd.Series(NOT_MISSING, index=series.index,
                              name=series.name)

        series_mv[series.isna()] = NOT_AVAILABLE

        print(series.name)

        return series_mv

    def _encode(self):
        super()._encode()

