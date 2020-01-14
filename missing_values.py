import pandas as pd
import numpy as np


def NHIS_heuristic(series):
    series_mv = pd.Series(0, index=series.index, name=series.name)

    # _, n_cols = df.shape

    # assert n_cols == 1

    val_max = series.max()

    if val_max < 10:  # Type 1 column
        series_mv[series == 7] = 2  # Type 2 missing values
        series_mv[series == 8] = 2  # Type 2 missing values
        series_mv[series == 9] = 2  # Type 2 missing values

    if val_max < 100:
        series_mv[series == 97] = 2  # Type 2 missing values
        series_mv[series == 98] = 2  # Type 2 missing values
        series_mv[series == 99] = 2  # Type 2 missing values

    series_mv[series.isna()] = 1  # Type 1 missing values

    return series_mv



def get_missing_values(df, heuristic):
    n_observations, n_features = df.shape
    columns = []

    for index in range(n_features):
        print(type(df.iloc[:, index]))
        columns.append(heuristic(df.iloc[:, index]))

    df_mv = pd.concat(columns, axis=1, sort=False)
    return df_mv


if __name__ == '__main__':
    data_folder = 'NHIS2017/data/'
    filename = data_folder + 'family/familyxx.csv'
    df = pd.read_csv(filename)

    print(get_missing_values(df, NHIS_heuristic))
