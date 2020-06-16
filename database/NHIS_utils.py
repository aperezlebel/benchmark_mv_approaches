"""Import NHIS csv data to Postgresql database."""
from dask import dataframe as dd
import pandas as pd
from sqlalchemy import create_engine

from features_type import _dump_feature_types
from .constants import ORDINAL, CONTINUE_R, CONTINUE_I, CATEGORICAL
from . import dbs


NHIS = dbs['NHIS']
translation = {
    'int32': CONTINUE_I,
    'int64': CONTINUE_I,
    'float64': CONTINUE_R,
    'object': ORDINAL,
}


def to_sql():
    """Import csv files to posgresql database."""
    engine = create_engine('postgresql://alexandreperez@localhost:5432/nhis')

    for name, path in NHIS.frame_paths.items():
        print(f'\n{name}\n\tReading csv')
        df = pd.read_csv(path)
        print(f'\tLowering columns')
        df.columns = [c.lower() for c in df.columns]
        print(f'\tConverting to sql')
        df.to_sql(name, engine, if_exists='replace')


def create_X_income(tables):
    """Create the X_income table."""
    print('\nX_income')
    household = tables['household']
    adult = tables['adult']
    family = tables['family']
    person = tables['person']

    print('\tMerging...')
    df = household.merge(
        family, how='inner', on=['SRVY_YR', 'HHX'],
        suffixes=('', '%to_drop')
    ).merge(
        person, how='inner', on=['SRVY_YR', 'HHX', 'FMX'],
        suffixes=('', '%to_drop')
    ).merge(
        adult, how='left', on=['SRVY_YR', 'HHX', 'FMX', 'FPX'],
        suffixes=('', '%to_drop')
    ).dropna(subset=['ERNYR_P']).compute()

    # Remove duplicate columns of the merges
    print('\tRemoving duplicates...')
    df = df.loc[:, ~df.columns.str.endswith('%to_drop')]

    # Rename columns having forbidden character _ (reserved for OH encoding)
    rename = {f: f.replace('_', '-') for f in df.columns}
    print(rename)
    df.rename(rename, axis=1, inplace=True)

    # Save index in columns
    df['IDX'] = df.index

    # Save new dataframe
    print('\tSaving...')
    df.to_csv(f'{NHIS.data_folder}custom/X_income.csv', index=None)

    # Compute the feature types and dump them
    _dump_types_X_income_v2(df)


def _dump_types_X_income_v1(df):
    """Create types using translation dict and dtypes."""
    print('\tDumping feature types...')
    types = {n: translation[str(t)] for n, t in df.dtypes.items()}
    types = pd.Series(types, index=df.columns)
    _dump_feature_types(types, NHIS, 'X_income', anonymize=False)


def _dump_types_X_income_v2(df):
    """Create types using files."""
    print('\tDumping feature types...')
    types = pd.Series(CATEGORICAL, index=df.columns)

    data_folder = NHIS.data_folder
    tables = ['household', 'family', 'person', 'adult']
    for table in tables:
        print(f'\t\t{table}')
        path = f'{data_folder}{table}/continue_i.txt'
        with open(path, 'r') as file:
            for line in file:
                line = line.rstrip()
                if line == '':
                    continue
                types[line.replace('_', '-')] = CONTINUE_I

    _dump_feature_types(types, NHIS, 'X_income', anonymize=False)


def create_table(table_name):
    """Create the asked table."""
    tables = {name: dd.read_csv(p) for name, p in NHIS.available_paths.items()}

    if table_name == 'X_income':
        create_X_income(tables)
