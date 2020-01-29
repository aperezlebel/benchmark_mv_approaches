"""Perform imputation on databases."""

import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

# from database import NHIS
# from missing_values import get_missing_values
from df_utils import fill_df, rint_features


def impute(df, df_mv, imputer):
    """Impute the missing value of the given DF using the given imputer.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame storing the input table with the missing data to impute.
    df_mv : pandas.DataFrame
        A data frame with same index and columns as df and storing boolean
        values encoding the presence of missing values. True: the cell is a
        missing value in df, False it is not.
    imputer : scikit learn imputer
        The imputer to use to impute the missing values of df. A copy of the
        imputer will be fitted on df. Note: The missing_values attribute of
        the imputer is ignored to detect the missing values.

    Returns
    -------
    pandas.DataFrame
        A copy of df with the missing values imputed using imputer.

    """
    imputer = deepcopy(imputer)
    mv_placeholder = -1

    # Find an unused placeholder for missing values
    while df.isin([mv_placeholder]).any().any():
        mv_placeholder = np.random.rand()

    # Replace the missing values with the placeholder
    df_aux = df.copy()
    fill_df(df_aux, df_mv, mv_placeholder)

    # Impute the missing values
    imputer.missing_values = mv_placeholder
    data_imputed = imputer.fit_transform(df_aux)

    del df_aux

    # Manage the case where an indicator was used to add binary columns for MV
    indicator = imputer.indicator_
    # If any, get feature ids for which an indicator column has been created
    features_with_mv = indicator.features_ if indicator is not None else []
    # If any, create names for these extra features.
    extra_columns = [f'indicator_{df.columns[id]}' for id in features_with_mv]

    return pd.DataFrame(data_imputed, index=df.index,
                        columns=list(df.columns)+extra_columns)


if __name__ == '__main__':
    from database import TB
    from database.constants import CATEGORICAL, CONTINUE_R
    import os

    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'Mean+mask': SimpleImputer(strategy='mean', add_indicator=True),
        'Med': SimpleImputer(strategy='median'),
        'Med+mask': SimpleImputer(strategy='median', add_indicator=True),
        'Iterative': IterativeImputer(),
        'Iterative+mask': IterativeImputer(add_indicator=True),
    }

    df = TB.encoded_dataframes['20000']
    # TB.encoded_dataframes['20000'].to_csv('sandbox/to_impute.csv',
    #                                   sep=';')
    # TB.encoded_missing_values['20000'].to_csv('sandbox/to_impute_mv.csv',
    #                                   sep=';')
    # df_mv = get_missing_values(df, NHIS.heuristic)
    df_mv = TB.encoded_missing_values['20000']

    os.makedirs('imputed/', exist_ok=True)

    for name, imputer in imputers.items():
        df_imputed = impute(df, df_mv != 0, imputer)
        print(df_imputed)
        df_imputed.to_csv(f'imputed/TB_20000_imputed_{name}.csv', sep=';')

        f_types = TB.encoded_feature_types['20000']
        df_rounded = rint_features(df_imputed, (f_types != CATEGORICAL) & (f_types != CONTINUE_R))
        df_rounded.to_csv(f'imputed/TB_20000_imputed_rounded_{name}.csv', sep=';')
