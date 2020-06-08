"""Gather all MIMIC related functions."""

import pandas as pd
import numpy as np

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING, \
    CATEGORICAL


class MIMIC(Database):

    def __init__(self, load=None):
        data_folder = None
        paths = {
        }
        sep = ','
        encoding = None
        encode = None

        super().__init__(
            name='MIMICIII',
            acronym='MIMIC',
            paths=paths,
            sep=sep,
            load=load,
            encoding=encoding,
            encode=encode)

    @staticmethod
    def heuristic(series):
        # The series storing the type of missing values
        series_mv = pd.Series(NOT_MISSING, index=series.index,
                              name=series.name)

        # series_mv[series.isna()] = NOT_AVAILABLE

        # print(series.name, end='\r')

        return series_mv

