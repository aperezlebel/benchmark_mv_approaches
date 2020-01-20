"""Load a couple of databases."""

import pandas as pd
from joblib import Memory

mem = Memory('cache_joblib')


@mem.cache
def load_NHIS():
    """Load the NHIS database."""
    data_folder = 'NHIS2017/data/'

    NHIS = {
        'family': pd.read_csv(data_folder+'family/familyxx.csv'),
        'child': pd.read_csv(data_folder+'child/samchild.csv'),
        'adult': pd.read_csv(data_folder+'adult/samadult.csv'),
        'person': pd.read_csv(data_folder+'person/personsx.csv'),
        'household': pd.read_csv(data_folder+'household/househld.csv'),
        'injury': pd.read_csv(data_folder+'injury/injpoiep.csv'),
    }

    return NHIS


NHIS = load_NHIS()


def load_TB():
    """Load the TraumaBase database."""
    data_folder = 'TraumaBase/'

    TB = {
        '20000': pd.read_csv(data_folder+'Traumabase_20000.csv', sep=';')
    }

    return TB


TB = load_TB()


if __name__ == '__main__':
    print(TB['20000'])
