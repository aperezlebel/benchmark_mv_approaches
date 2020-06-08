"""Check data types and create feature types file for MIMIC."""
import pandas as pd

from features_type import _dump_feature_types
from database import dbs
from .constants import CONTINUE_R, CONTINUE_I, BINARY, CATEGORICAL, \
    NOT_A_FEATURE, DATE_EXPLODED

translation = {
    'integer': CONTINUE_I,
    'smallint': CONTINUE_I,
    'double precision': CONTINUE_R,
    'timestamp without time zone': DATE_EXPLODED,
    'character varying': NOT_A_FEATURE,
    'character': NOT_A_FEATURE,
    'text': NOT_A_FEATURE,
}


def check_data_types(filepath):
    """Check data types."""
    df = pd.read_csv(filepath)

    data_types = dict()

    for _, row in df.iterrows():
        table_name, column_name, data_type = row

        existing_data_type = data_types.get(column_name, None)
        if existing_data_type is None:
            data_types[column_name] = data_type
            continue

        elif existing_data_type == data_type:
            continue

        print(
            f'\nFound different data types for same column name.\n'
            f'Column: {column_name}\n'
            f'Type 1: {existing_data_type}\n'
            f'Type 2: {data_type}\n'
            f'Table: {table_name}\n'
        )


def MIMIC_feature_types(filepath):
    df = pd.read_csv(filepath)

    types = dict()

    for _, row in df.iterrows():
        table_name, column_name, data_type = row

        types[f'{table_name}/{column_name}'] = translation[data_type]

    types = pd.Series(types)
    _dump_feature_types(types, dbs['MIMIC'], 'mimiciii', anonymize=False)
    print(types)


if __name__ == '__main__':
    check_data_types('/Users/alexandreperez/OneDrive/Documents/Stage/MILA/Task3_NHIS/NHIS_analyse/MIMICIII/physionet.org/files/mimiciii/data_types.csv')
    MIMIC_feature_types('/Users/alexandreperez/OneDrive/Documents/Stage/MILA/Task3_NHIS/NHIS_analyse/MIMICIII/physionet.org/files/mimiciii/data_types.csv')
