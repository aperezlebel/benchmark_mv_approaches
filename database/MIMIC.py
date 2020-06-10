"""Gather all MIMIC related functions."""

import pandas as pd
import numpy as np

from .base import Database
from .constants import NOT_APPLICABLE, NOT_AVAILABLE, NOT_MISSING, \
    CATEGORICAL


class MIMIC(Database):

    def __init__(self, load=None):
        data_folder = 'MIMICIII/physionet.org/files/mimiciii/1.4/'
        paths = {
            'procedureevents_mv': f'{data_folder}PROCEDUREEVENTS_MV.csv',
            'callout': f'{data_folder}CALLOUT.csv',
            'd_cpt': f'{data_folder}D_CPT.csv',
            'd_items': f'{data_folder}D_ITEMS.csv',
            'caregivers': f'{data_folder}CAREGIVERS.csv',
            'microbiologyevents': f'{data_folder}MICROBIOLOGYEVENTS.csv',
            'labevents': f'{data_folder}LABEVENTS.csv',
            'inputevents_cv': f'{data_folder}INPUTEVENTS_CV.csv',
            'admissions': f'{data_folder}ADMISSIONS.csv',
            'd_labitems': f'{data_folder}D_LABITEMS.csv',
            'datetimeevents': f'{data_folder}DATETIMEEVENTS.csv',
            'prescriptions': f'{data_folder}PRESCRIPTIONS.csv',
            'procedures_icd': f'{data_folder}PROCEDURES_ICD.csv',
            'noteevents': f'{data_folder}NOTEEVENTS.csv',
            'chartevents': f'{data_folder}CHARTEVENTS.csv',
            'transfers': f'{data_folder}TRANSFERS.csv',
            'diagnoses_icd': f'{data_folder}DIAGNOSES_ICD.csv',
            'services': f'{data_folder}SERVICES.csv',
            'drgcodes': f'{data_folder}DRGCODES.csv',
            'outputevents': f'{data_folder}OUTPUTEVENTS.csv',
            'patients': f'{data_folder}PATIENTS.csv',
            'd_icd_diagnoses': f'{data_folder}D_ICD_DIAGNOSES.csv',
            'icustays': f'{data_folder}ICUSTAYS.csv',
            'inputevents_mv': f'{data_folder}INPUTEVENTS_MV.csv',
            'd_icd_procedures': f'{data_folder}D_ICD_PROCEDURES.csv',
            'cptevents': f'{data_folder}CPTEVENTS.csv',
            'X_labevents': f'{data_folder}custom/X_labevents.csv',
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

