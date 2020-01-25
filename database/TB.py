"""Gather all TraumaBase related functions."""

import pandas as pd

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING


class _TB(Database):

    def __init__(self):
        super().__init__('TraumaBase', 'TB')
        self._encode()

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
    def heuristic(series):
        # The series storing the type of missing values
        series_mv = pd.Series(NOT_MISSING, index=series.index,
                              name=series.name)

        series_mv[series.isna()] = NOT_AVAILABLE
        series_mv[series == 'NA'] = NOT_AVAILABLE
        series_mv[series == 'ND'] = NOT_AVAILABLE
        series_mv[series == 'NR'] = NOT_AVAILABLE
        series_mv[series == 'NF'] = NOT_AVAILABLE
        series_mv[series == 'NDC'] = NOT_AVAILABLE
        series_mv[series == 'IMP'] = NOT_AVAILABLE

        print(series.name)

        if series.name == 'PaO2/FIO2 (mmHg) si VM ou CPAP':
            series_mv[series == 'Non applicable :  ni VM ni CPAP'] = NOT_APPLICABLE

        if series.name == 'Glasgow':
            series_mv[series == '06/09/2019 00:00'] = NOT_AVAILABLE
            series_mv[series == '10/12/2019 00:00'] = NOT_AVAILABLE

        if series.name == 'CGR 24h':
            series_mv[series == 'Pas de choc hémorragique'] = NOT_APPLICABLE

        if series.name == 'Pression intracrânienne (PIC)':
            series_mv[series == 'Pas de TC'] = NOT_APPLICABLE

        return series_mv


TB = _TB()
