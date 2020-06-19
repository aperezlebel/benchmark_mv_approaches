"""Gather all TraumaBase related functions."""

import pandas as pd
import numpy as np
import os

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING


class TB(Database):

    def __init__(self, load=None):

        data_folder = 'TraumaBase/'
        paths = {
            '20000': data_folder+'Traumabase_20000.csv'
        }
        sep = ';'
        encode = 'all'

        super().__init__(
            name='TraumaBase',
            acronym='TB',
            paths=paths,
            sep=sep,
            load=load,
            encode=encode
            )

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

        # print(series.name, end='\r')

        if series.name == 'PaO2/FIO2 (mmHg) si VM ou CPAP':
            series_mv[series == 'Non applicable :  ni VM ni CPAP'] = NOT_APPLICABLE

        if series.name == 'Glasgow':
            series_mv[series == '06/09/2019 00:00'] = NOT_AVAILABLE
            series_mv[series == '10/12/2019 00:00'] = NOT_AVAILABLE

        if series.name == 'CGR 24h':
            series_mv[series == 'Pas de choc hémorragique'] = NOT_APPLICABLE

        if series.name == 'Pression intracrânienne (PIC)':
            series_mv[series == 'Pas de TC'] = NOT_APPLICABLE

        if series.name == 'Nombre de pneumopathies':
            series_mv[series == 'Non'] = NOT_APPLICABLE
            series_mv[series == 'Oui'] = NOT_AVAILABLE

        if series.name == 'Jour de la première pneumopathie':
            series_mv[series == 'Non'] = NOT_APPLICABLE
            series_mv[series == 'Oui'] = NOT_AVAILABLE

        if series.name == 'Régression mydriase sous osmothérapie':
            series_mv[series == 'Non testé'] = NOT_AVAILABLE

        if series.name == 'Lieu du traumatisme':
            series_mv[series == 'Non-spécifié'] = NOT_AVAILABLE

        if series.name == 'DATE_ENTREE':
            series_mv[series == np.nan] = NOT_AVAILABLE

        if series.name == 'Dose noradrénaline au moment départ au scan':
            series_mv[series == 'Rien'] = NOT_AVAILABLE

        return series_mv

