"""Gather all TraumaBase related functions."""

import pandas as pd

from .base import Database


class _TB(Database):

    def __init__(self):
        super().__init__()
        self.name = 'TraumaBase'
        self.acronym = 'TB'
        self._load_feature_types()

    def _load_db(self, encode=True):
        """Load the TraumaBase database."""
        data_folder = 'TraumaBase/'

        self.dataframes = {
            '20000': pd.read_csv(data_folder+'Traumabase_20000.csv', sep=';')
        }

        # if encode:
        #     df = TB['20000'].copy()
        #     df_mv = get_missing_values(df, TB_heuristic)
        #     df.where(df_mv == 0, other=-1, inplace=True)
        #     df_encoded, _ = ordinal_encode(df)
        #     df_encoded[df_mv != 0] = np.nan
        #     TB['20000'] = df_encoded

    @staticmethod
    def heuristic(self, series):
        # The series storing the type of missing values
        series_mv = pd.Series(0, index=series.index, name=series.name)

        series_mv[series.isna()] = 2
        series_mv[series == 'NA'] = 2
        series_mv[series == 'ND'] = 2

        return series_mv


TB = _TB()
