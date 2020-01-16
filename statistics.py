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
    # 1: Statistics on the full database
    n_rows, n_cols = df_mv.shape
    n_values = n_rows*n_cols
    df_mv1 = df_mv == 1
    df_mv2 = df_mv == 2

    # Number of missing values in the DB
    n_mv1 = df_mv1.values.sum()
    n_mv2 = df_mv2.values.sum()
    n_mv = n_mv1 + n_mv2
    n_not_mv = n_values - n_mv

    # Frequencies of missing values in the DB
    f_mv1 = 100*n_mv1/n_values
    f_mv2 = 100*n_mv2/n_values
    f_mv = 100*n_mv/n_values
    f_not_mv = 100*n_not_mv/n_values

    # Print these statistics
    print(
        f'\n'
        f'Statistics on the full database:\n'
        f'---------------------------------\n'
        f'[{n_rows} rows x {n_cols} columns]\n'
        f'{n_values} values\n'
        f'N NMV:    {f_not_mv:.1f}% or {n_not_mv}\n'
        f'N MV:     {f_mv:.1f}% or {n_mv}\n'
        f'    N MV 1:   {f_mv1:.1f}% or {n_mv1}\n'
        f'    N MV 2:   {f_mv2:.1f}% or {n_mv2}\n'
    )

    # If asked, plot these statistics
    if show:
        _, ax = plt.subplots(figsize=(10, 4))

        df_show = pd.DataFrame({
            'MV1': [n_mv1],
            'MV2': [n_mv2],
            'MV': [n_mv],
            'V': [n_values],
            'type': ['Full database']
            })

        sns.set_color_codes('pastel')
        sns.barplot(x='V', y='type', data=df_show, color='b',
                    label=f'Not missing ({f_not_mv:.1f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='MV', y='type', data=df_show, color='b',
                    label=f'Missing - Not applicable ({f_mv1:.1f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='MV2', y='type', data=df_show, color='b',
                    label=f'Missing - Not available ({f_mv2:.1f}%)')

        ax.legend(ncol=1, loc='lower right', frameon=True)
        ax.set(ylabel='', xlabel='Number of values')
        sns.despine(left=True, bottom=True)

    # 2: Statistics feature-wise
    n_mv1_fw = df_mv1.sum().to_frame('N MV1')  # Number of MV 1 by feature
    n_mv2_fw = df_mv2.sum().to_frame('N MV2')  # Number of MV 2 by feature

    n_mv_fw = pd.concat([n_mv1_fw, n_mv2_fw], axis=1)
    n_mv_fw['N MV'] = n_mv_fw['N MV1'] + n_mv_fw['N MV2']

    # Sort by number of missing values
    n_mv_fw.sort_values('N MV', ascending=False, inplace=True)

    # Number of features with missing values
    df_mv_fw = (n_mv_fw != 0).sum()
    n_f_mv1 = df_mv_fw['N MV1']
    n_f_mv2 = df_mv_fw['N MV2']
    n_f_mv_1o2 = df_mv_fw['N MV']
    n_f_mv_1a2 = n_f_mv1 + n_f_mv2 - n_f_mv_1o2
    n_f_mv1_o = n_f_mv1 - n_f_mv_1a2
    n_f_mv2_o = n_f_mv2 - n_f_mv_1a2

    # Frequencies of features with missing values
    f_f_mv = 100*n_f_mv_1o2/n_cols
    f_f_mv1_o = 100*n_f_mv1_o/n_cols
    f_f_mv2_o = 100*n_f_mv2_o/n_cols
    f_f_mv_1a2 = 100*n_f_mv_1a2/n_cols

    with pd.option_context('display.max_rows', None):
        print(
            f'\n'
            f'Statistics feature-wise:\n'
            f'------------------------\n'
            f'N F MV:             {n_f_mv_1o2} ({f_f_mv:.1f}%)\n'
            f'    N F MV1 only:   {n_f_mv1_o} ({f_f_mv1_o:.1f}%)\n'
            f'    N F MV2 only:   {n_f_mv2_o} ({f_f_mv2_o:.1f}%)\n'
            f'    N F MV 1 and 2: {n_f_mv_1a2} ({f_f_mv_1a2:.1f}%)\n'
            f'\n'
            f'{n_mv_fw}'
        )

    if show:
        # Copy index in a column for the barplot method
        n_mv_fw['feature'] = n_mv_fw.index

        # Add the total number of values for each feature
        n_mv_fw['N V'] = n_rows

        # Get rid of the features with no missing values
        n_mv_fw_l = n_mv_fw[(n_mv_fw['N MV1'] != 0) | (n_mv_fw['N MV2'] != 0)]

        _, ax = plt.subplots(figsize=(8, 8))

        sns.set_color_codes('pastel')
        sns.barplot(x='N V', y='feature', data=n_mv_fw_l,
                    color='b', label=f'Not missing')

        sns.set_color_codes('muted')
        sns.barplot(x='N MV', y='feature', data=n_mv_fw_l,
                    color='b', label=f'Missing - Not applicable')

        sns.set_color_codes("dark")
        sns.barplot(x='N MV2', y='feature', data=n_mv_fw_l,
                    color="b", label=f'Missing - Not available')

        ax.legend(ncol=1, loc='lower right', frameon=True)
        ax.set(ylabel='', xlabel='Number of values')
        ax.tick_params(labelsize=5)
        sns.despine(left=True, bottom=True)

    # 3: Rows without missing values
    # Number
    n_r_w_mv1 = df_mv1.any(axis=1).sum()
    n_r_w_mv2 = df_mv2.any(axis=1).sum()
    n_r_w_mv = df_mv.any(axis=1).sum()

    # Frequencies
    f_r_w_mv1 = 100*n_r_w_mv1/n_rows
    f_r_w_mv2 = 100*n_r_w_mv2/n_rows
    f_r_w_mv = 100*n_r_w_mv/n_rows

    print(
        f'\n'
        f'Statistics on rows:\n'
        f'-------------------\n'
        f'N rows: {n_rows}\n'
        f'N rows with MV1: {n_r_w_mv1} ({f_r_w_mv1:.2f}%)\n'
        f'N rows with MV2: {n_r_w_mv2} ({f_r_w_mv2:.2f}%)\n'
        f'N rows with MV:  {n_r_w_mv} ({f_r_w_mv:.2f}%)\n'
    )

    plt.show()


if __name__ == '__main__':
    df = NHIS['family']
    df_mv = get_missing_values(df, NHIS_heuristic)

    describe_missing_values(df_mv, show=True)
