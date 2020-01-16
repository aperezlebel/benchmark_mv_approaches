"""Compute statistics about missing values on a databse."""

import pandas as pd
import matplotlib
matplotlib.use('MacOSX')
import seaborn as sns
import matplotlib.pyplot as plt

from database import NHIS
from heuristic import NHIS as NHIS_heuristic
from missing_values import get_missing_values


def describe_missing_values(df_mv, show=False):
    n_rows, n_cols = df_mv.shape
    n_values = n_rows*n_cols
    df_mv_type1 = df_mv == 1
    df_mv_type2 = df_mv == 2

    # Number of missing values in the DB
    n_mv_1 = df_mv_type1.values.sum()
    n_mv_2 = df_mv_type2.values.sum()
    n_mv = n_mv_1 + n_mv_2
    n_not_mv = n_values - n_mv

    # Frequencies of missing values in the DB
    f_mv_1 = 100*n_mv_1/n_values
    f_mv_2 = 100*n_mv_2/n_values
    f_mv = 100*n_mv/n_values
    f_not_mv = 100*n_not_mv/n_values

    # Print these statistics
    print(
        f'\n'
        f'Statistic on the full database:\n'
        f'---------------------------------\n'
        f'[{n_rows} rows x {n_cols} columns]\n'
        f'{n_values} values\n'
        f'N NMV:    {f_not_mv:.1f}% or {n_not_mv}\n'
        f'N MV:     {f_mv:.1f}% or {n_mv}\n'
        f'    N MV 1:   {f_mv_1:.1f}% or {n_mv_1}\n'
        f'    N MV 2:   {f_mv_2:.1f}% or {n_mv_2}\n'
        )

    # If asked, plot these statistics
    if show:
        _, ax = plt.subplots(figsize=(10, 4))

        df_show = pd.DataFrame({
            'MV1': [n_mv_1],
            'MV2': [n_mv_2],
            'MV': [n_mv],
            'V': [n_values],
            'type': ['Full database']
            })

        sns.set_color_codes('pastel')
        sns.barplot(x='V', y='type', data=df_show,
                    color='b', label=f'Not missing ({f_not_mv:.1f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='MV', y='type', data=df_show,
                    color='b', label=f'Type 1 ({f_mv_1:.1f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='MV2', y='type', data=df_show,
                    color='b', label=f'Type 2 ({f_mv_2:.1f}%)')

        ax.legend(ncol=3, loc='lower right', frameon=True)
        ax.set(ylabel='',
               xlabel='Number of missing values')
        sns.despine(left=True, bottom=True)

    plt.show()


if __name__ == '__main__':
    df = NHIS['family']
    df_mv = get_missing_values(df, NHIS_heuristic)

    describe_missing_values(df_mv, show=False)
