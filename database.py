"""Load a couple of databases."""

import pandas as pd
import numpy as np
from joblib import Memory

from encode import ordinal_encode
from heuristic import NHIS as NHIS_heuristic
from heuristic import TB as TB_heuristic
from missing_values import get_missing_values

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


def load_TB(encode=True):
    """Load the TraumaBase database."""
    data_folder = 'TraumaBase/'

    TB = {
        '20000': pd.read_csv(data_folder+'Traumabase_20000.csv', sep=';')
    }

    # if encode:
    #     df = TB['20000'].copy()
    #     df_mv = get_missing_values(df, TB_heuristic)
    #     df.where(df_mv == 0, other=-1, inplace=True)
    #     df_encoded, _ = ordinal_encode(df)
    #     df_encoded[df_mv != 0] = np.nan
    #     TB['20000'] = df_encoded

    return TB


TB = load_TB(encode=True)


if __name__ == '__main__':
    print(TB['20000'])
