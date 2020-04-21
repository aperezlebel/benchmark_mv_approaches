"""Compute statistics about missing values on a databse."""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('MacOSX')
import seaborn as sns
import matplotlib.pyplot as plt

from missing_values import get_missing_values
from prediction.tasks import tasks
from database import dbs


def get_indicators_mv(df_mv):
    # 1: Statistics on the full database
    n_rows, n_cols = df_mv.shape
    n_values = n_rows*n_cols
    df_mv1 = df_mv == 1
    df_mv2 = df_mv == 2
    df_mv_bool = df_mv != 0

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

    # Store the indicators in a df
    df_1 = pd.DataFrame({
        'n_rows': [n_rows],
        'n_cols': [n_cols],
        'n_values': [n_values],
        'n_mv': [n_mv],
        'n_mv1': [n_mv1],
        'n_mv2': [n_mv2],
        'n_not_mv': [n_not_mv],
        'f_mv': [f_mv],
        'f_mv1': [f_mv1],
        'f_mv2': [f_mv2],
        'f_not_mv': [f_not_mv],
    })

    # 2: Number of features with missing values
    # For each feature, tells if it contains MV of type 1
    df_f_w_mv1 = df_mv1.any().rename('MV1')
    # For each feature, tells if it contains MV of type 2
    df_f_w_mv2 = df_mv2.any().rename('MV2')
    # Concat previous series
    df_f_w_mv = pd.concat([df_f_w_mv1, df_f_w_mv2], axis=1)

    # Add columns for logical combination of the two series
    df_f_w_mv['MV'] = df_f_w_mv['MV1'] | df_f_w_mv['MV2']  # MV1 or MV2
    df_f_w_mv['MV1a2'] = df_f_w_mv['MV1'] & df_f_w_mv['MV2']  # MV1 and MV2
    df_f_w_mv['MV1o'] = df_f_w_mv['MV1'] & ~df_f_w_mv['MV2']  # MV1 only
    df_f_w_mv['MV2o'] = ~df_f_w_mv['MV1'] & df_f_w_mv['MV2']  # MV2 only

    # By summing, derive the number of features with MV of a given type
    df_n_f_w_mv = df_f_w_mv.sum()

    # Numbers of features with missing values
    n_f_w_mv = df_n_f_w_mv['MV']  # MV1 or MV2
    n_f_w_mv1_o = df_n_f_w_mv['MV1o']  # MV1 only
    n_f_w_mv2_o = df_n_f_w_mv['MV2o']  # MV2 only
    n_f_w_mv_1a2 = df_n_f_w_mv['MV1a2']  # MV1 and MV2
    n_f_wo_mv = n_cols - df_n_f_w_mv['MV']  # Without MV

    # Frequencies of features with missing values
    f_f_w_mv1_o = 100*n_f_w_mv1_o/n_cols
    f_f_w_mv2_o = 100*n_f_w_mv2_o/n_cols
    f_f_w_mv = 100*n_f_w_mv/n_cols
    f_f_w_mv_1a2 = 100*n_f_w_mv_1a2/n_cols
    f_f_wo_mv = 100*n_f_wo_mv/n_cols

    # Store the indicators in a df
    df_2 = pd.DataFrame({
        'n_f_w_mv': [n_f_w_mv],
        'n_f_w_mv1_o': [n_f_w_mv1_o],
        'n_f_w_mv2_o': [n_f_w_mv2_o],
        'n_f_w_mv_1a2': [n_f_w_mv_1a2],
        'n_f_wo_mv': [n_f_wo_mv],
        'f_f_w_mv': [f_f_w_mv],
        'f_f_w_mv1_o': [f_f_w_mv1_o],
        'f_f_w_mv2_o': [f_f_w_mv2_o],
        'f_f_w_mv_1a2': [f_f_w_mv_1a2],
        'f_f_wo_mv': [f_f_wo_mv],
    })

    # 3: Statistics feature-wise
    n_mv1_fw = df_mv1.sum().to_frame('N MV1')  # Number of MV 1 by feature
    n_mv2_fw = df_mv2.sum().to_frame('N MV2')  # Number of MV 2 by feature

    n_mv_fw = pd.concat([n_mv1_fw, n_mv2_fw], axis=1)
    n_mv_fw['N MV'] = n_mv_fw['N MV1'] + n_mv_fw['N MV2']
    n_mv_fw['F MV1'] = 100*n_mv_fw['N MV1']/n_rows
    n_mv_fw['F MV2'] = 100*n_mv_fw['N MV2']/n_rows
    n_mv_fw['F MV'] = 100*n_mv_fw['N MV']/n_rows

    # Sort by number of missing values
    n_mv_fw.sort_values('N MV', ascending=False, inplace=True)

    # Store the indicators in a df
    df_3 = n_mv_fw

    # 4: Rows without missing values
    # For each row, tells if it contains MV of type 1
    df_r_w_mv1 = df_mv1.any(axis=1).rename('MV1')
    # For each row, tells if it contains MV of type 2
    df_r_w_mv2 = df_mv2.any(axis=1).rename('MV2')
    # Concat previous series
    df_r_w_mv = pd.concat([df_r_w_mv1, df_r_w_mv2], axis=1)

    # Add columns for logical combination of the two series
    df_r_w_mv['MV'] = df_r_w_mv['MV1'] | df_r_w_mv['MV2']  # MV1 or MV2
    df_r_w_mv['MV1a2'] = df_r_w_mv['MV1'] & df_r_w_mv['MV2']  # MV1 and MV2
    df_r_w_mv['MV1o'] = df_r_w_mv['MV1'] & ~df_r_w_mv['MV2']  # MV1 only
    df_r_w_mv['MV2o'] = ~df_r_w_mv['MV1'] & df_r_w_mv['MV2']  # MV2 only

    # By summing, derive the number of rows with MV of a given type
    df_n_r_w_mv = df_r_w_mv.sum()

    # Numbers of rows with missing values
    n_r_w_mv = df_n_r_w_mv['MV']  # MV1 or MV2
    n_r_w_mv1_o = df_n_r_w_mv['MV1o']  # MV1 only
    n_r_w_mv2_o = df_n_r_w_mv['MV2o']  # MV2 only
    n_r_w_mv_1a2 = df_n_r_w_mv['MV1a2']  # MV1 and MV2
    n_r_wo_mv = n_rows - df_n_r_w_mv['MV']  # Without MV

    # Frequencies of rows with missing values
    f_r_w_mv1_o = 100*n_r_w_mv1_o/n_rows
    f_r_w_mv2_o = 100*n_r_w_mv2_o/n_rows
    f_r_w_mv = 100*n_r_w_mv/n_rows
    f_r_w_mv_1a2 = 100*n_r_w_mv_1a2/n_rows
    f_r_wo_mv = 100*n_r_wo_mv/n_rows

    # Store the indicators in a df
    df_4 = pd.DataFrame({
        'n_r_w_mv': [n_r_w_mv],
        'n_r_w_mv1_o': [n_r_w_mv1_o],
        'n_r_w_mv2_o': [n_r_w_mv2_o],
        'n_r_w_mv_1a2': [n_r_w_mv_1a2],
        'n_r_wo_mv': [n_r_wo_mv],
        'f_r_w_mv': [f_r_w_mv],
        'f_r_w_mv1_o': [f_r_w_mv1_o],
        'f_r_w_mv2_o': [f_r_w_mv2_o],
        'f_r_w_mv_1a2': [f_r_w_mv_1a2],
        'f_r_wo_mv': [f_r_wo_mv],
    })

    # 5: Number of rows affected if we remove features with MV
    df_f_w_mv1 = df_f_w_mv['MV1']  # Series of features having MV1
    df_f_w_mv2 = df_f_w_mv['MV2']  # Series of features having MV2
    df_f_w_mv_1o2 = df_f_w_mv['MV']  # Series of features having MV1 or MV2
    df_f_w_mv1_o = df_f_w_mv['MV1o']  # Series of features having MV1 only
    df_f_w_mv2_o = df_f_w_mv['MV2o']  # Series of features having MV2 only
    df_f_w_mv_1a2 = df_f_w_mv['MV1a2']  # Series of features having MV1 and MV2

    df_features = pd.Series(True, index=df_f_w_mv.index)

    features_to_drop_mv1 = df_features.loc[~df_f_w_mv1].index
    features_to_drop_mv2 = df_features.loc[~df_f_w_mv2].index
    features_to_drop_mv_1o2 = df_features.loc[~df_f_w_mv_1o2].index
    features_to_drop_mv1_o = df_features.loc[~df_f_w_mv1_o].index
    features_to_drop_mv2_o = df_features.loc[~df_f_w_mv2_o].index
    features_to_drop_mv_1a2 = df_features.loc[~df_f_w_mv_1a2].index

    df_mv1_dropped = df_mv_bool.drop(features_to_drop_mv1, 1)
    df_mv2_dropped = df_mv_bool.drop(features_to_drop_mv2, 1)
    df_mv_1o2_dropped = df_mv_bool.drop(features_to_drop_mv_1o2, 1)
    df_mv1_o_dropped = df_mv_bool.drop(features_to_drop_mv1_o, 1)
    df_mv2_o_dropped = df_mv_bool.drop(features_to_drop_mv2_o, 1)
    df_mv_1a2_dropped = df_mv_bool.drop(features_to_drop_mv_1a2, 1)

    # Number of rows affected if we remove feature having MV of type:
    n_r_a_rm_mv1 = (~df_mv1_dropped).any(axis=1).sum()  # MV1
    n_r_a_rm_mv2 = (~df_mv2_dropped).any(axis=1).sum()  # MV2
    n_r_a_rm_mv_1o2 = (~df_mv_1o2_dropped).any(axis=1).sum()  # MV1 or MV2
    n_r_a_rm_mv1_o = (~df_mv1_o_dropped).any(axis=1).sum()  # MV1 only
    n_r_a_rm_mv2_o = (~df_mv2_o_dropped).any(axis=1).sum()  # MV2 only
    n_r_a_rm_mv_1a2 = (~df_mv_1a2_dropped).any(axis=1).sum()  # MV1 and MV2

    # Frequencies of rows affected if we remove feature having MV of type:
    f_r_a_rm_mv1 = 100*n_r_a_rm_mv1/n_rows  # MV1
    f_r_a_rm_mv2 = 100*n_r_a_rm_mv2/n_rows  # MV2
    f_r_a_rm_mv_1o2 = 100*n_r_a_rm_mv_1o2/n_rows  # MV1 or MV2
    f_r_a_rm_mv1_o = 100*n_r_a_rm_mv1_o/n_rows  # MV1 only
    f_r_a_rm_mv2_o = 100*n_r_a_rm_mv2_o/n_rows  # MV2 only
    f_r_a_rm_mv_1a2 = 100*n_r_a_rm_mv_1a2/n_rows  # MV1 and MV2

    # Store the indicators in a df
    df_5 = pd.DataFrame({
        'n_r_a_rm_mv1': [n_r_a_rm_mv1],
        'n_r_a_rm_mv2': [n_r_a_rm_mv2],
        'n_r_a_rm_mv_1o2': [n_r_a_rm_mv_1o2],
        'n_r_a_rm_mv1_o': [n_r_a_rm_mv1_o],
        'n_r_a_rm_mv2_o': [n_r_a_rm_mv2_o],
        'n_r_a_rm_mv_1a2': [n_r_a_rm_mv_1a2],
        'f_r_a_rm_mv1': [f_r_a_rm_mv1],
        'f_r_a_rm_mv2': [f_r_a_rm_mv2],
        'f_r_a_rm_mv_1o2': [f_r_a_rm_mv_1o2],
        'f_r_a_rm_mv1_o': [f_r_a_rm_mv1_o],
        'f_r_a_rm_mv2_o': [f_r_a_rm_mv2_o],
        'f_r_a_rm_mv_1a2': [f_r_a_rm_mv_1a2],
    })

    # 6: Proportion of information lost when removing features with MV
    # Number
    n_v_lost_mv1 = (~df_mv1_dropped).sum().sum()
    n_v_lost_mv2 = (~df_mv2_dropped).sum().sum()
    n_v_lost_mv_1o2 = (~df_mv_1o2_dropped).sum().sum()
    n_v_lost_mv1_o = (~df_mv1_o_dropped).sum().sum()
    n_v_lost_mv2_o = (~df_mv2_o_dropped).sum().sum()
    n_v_lost_mv_1a2 = (~df_mv_1a2_dropped).sum().sum()

    # Frequencies
    f_v_lost_mv1 = 100*n_v_lost_mv1/n_values
    f_v_lost_mv2 = 100*n_v_lost_mv2/n_values
    f_v_lost_mv_1o2 = 100*n_v_lost_mv_1o2/n_values
    f_v_lost_mv1_o = 100*n_v_lost_mv1_o/n_values
    f_v_lost_mv2_o = 100*n_v_lost_mv2_o/n_values
    f_v_lost_mv_1a2 = 100*n_v_lost_mv_1a2/n_values

    # Store the indicators in a df
    df_6 = pd.DataFrame({
        'n_v_lost_mv1': [n_v_lost_mv1],
        'n_v_lost_mv2': [n_v_lost_mv2],
        'n_v_lost_mv_1o2': [n_v_lost_mv_1o2],
        'n_v_lost_mv1_o': [n_v_lost_mv1_o],
        'n_v_lost_mv2_o': [n_v_lost_mv2_o],
        'n_v_lost_mv_1a2': [n_v_lost_mv_1a2],
        'f_v_lost_mv1': [f_v_lost_mv1],
        'f_v_lost_mv2': [f_v_lost_mv2],
        'f_v_lost_mv_1o2': [f_v_lost_mv_1o2],
        'f_v_lost_mv1_o': [f_v_lost_mv1_o],
        'f_v_lost_mv2_o': [f_v_lost_mv2_o],
        'f_v_lost_mv_1a2': [f_v_lost_mv_1a2],
    })

    return {
        'global': df_1,
        'features': df_2,
        'feature-wise': df_3,
        'rows': df_4,
        'rm_rows': df_5,
        'rm_features': df_6,
    }


# Plot functions: each indicator has a differebt way of beaing plotted
def plot_global(indicators, plot=False, show=True, ax=None):
    """Plot statistics on the full database."""
    # Get required indicators
    df = indicators['global']

    n_rows = df.at[0, 'n_rows']
    n_cols = df.at[0, 'n_cols']
    n_values = df.at[0, 'n_values']
    n_mv = df.at[0, 'n_mv']
    n_mv1 = df.at[0, 'n_mv1']
    n_mv2 = df.at[0, 'n_mv2']
    n_not_mv = df.at[0, 'n_not_mv']
    f_mv = df.at[0, 'f_mv']
    f_mv1 = df.at[0, 'f_mv1']
    f_mv2 = df.at[0, 'f_mv2']
    f_not_mv = df.at[0, 'f_not_mv']

    # Print these statistics
    if show:
        print(
            f'\n'
            f'Statistics on the full data frame:\n'
            f'---------------------------------\n'
            f'[{n_rows} rows x {n_cols} columns]\n'
            f'{n_values} values\n'
            f'N NMV:    {f_not_mv:.1f}% or {n_not_mv}\n'
            f'N MV:     {f_mv:.1f}% or {n_mv}\n'
            f'    N MV 1:   {f_mv1:.1f}% or {n_mv1}\n'
            f'    N MV 2:   {f_mv2:.1f}% or {n_mv2}\n'
        )

    # If asked, plot these statistics
    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        df_show = pd.DataFrame({
            'MV1': [n_mv1],
            'MV2': [n_mv2],
            'MV': [n_mv],
            'V': [n_values],
            'type': ['Full data frame']
            })

        sns.set_color_codes('pastel')
        sns.barplot(x='V', y='type', data=df_show, color='b', ax=ax,
                    label=f'Not missing ({f_not_mv:.1f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='MV', y='type', data=df_show, color='b', ax=ax,
                    label=f'Missing - Not applicable ({f_mv1:.1f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='MV2', y='type', data=df_show, color='b', ax=ax,
                    label=f'Missing - Not available ({f_mv2:.1f}%)')

        ax.legend(ncol=1, loc='lower right', frameon=True,
                  title='Type of values')
        ax.set(ylabel='', xlabel=f'Number of values (Total {n_values})')
        sns.despine(left=True, bottom=True, ax=ax)


def plot_features(indicators, plot=False, show=True, ax=None):
    """Plot the number of features with missing values."""
    # Get required indicators
    df = pd.concat([indicators['features'], indicators['global']], axis=1)

    n_f_w_mv = df.at[0, 'n_f_w_mv']
    n_f_w_mv1_o = df.at[0, 'n_f_w_mv1_o']
    n_f_w_mv2_o = df.at[0, 'n_f_w_mv2_o']
    n_f_w_mv_1a2 = df.at[0, 'n_f_w_mv_1a2']
    n_f_wo_mv = df.at[0, 'n_f_wo_mv']
    f_f_w_mv = df.at[0, 'f_f_w_mv']
    f_f_w_mv1_o = df.at[0, 'f_f_w_mv1_o']
    f_f_w_mv2_o = df.at[0, 'f_f_w_mv2_o']
    f_f_w_mv_1a2 = df.at[0, 'f_f_w_mv_1a2']
    f_f_wo_mv = df.at[0, 'f_f_wo_mv']

    n_cols = df.at[0, 'n_cols']

    if show:
        print(
            f'\n'
            f'Statistics on features:\n'
            f'-----------------------\n'
            f'N features: {n_cols}\n'
            f'N features with MV:              {n_f_w_mv} ({f_f_w_mv:.1f}%)\n'
            f'    N features with MV1 only:    {n_f_w_mv1_o} ({f_f_w_mv1_o:.1f}%)\n'
            f'    N features with MV2 only:    {n_f_w_mv2_o} ({f_f_w_mv2_o:.1f}%)\n'
            f'    N features with MV1 and MV2: {n_f_w_mv_1a2} ({f_f_w_mv_1a2:.1f}%)\n'
        )

    if plot:
        # Plot proportion of features with missing values
        df_show = pd.DataFrame({
            'N MV': [n_f_w_mv],
            'N MV1 only': [n_f_w_mv1_o],
            'N MV2 only': [n_f_w_mv2_o],
            'N MV 1 xor 2': [n_f_w_mv1_o + n_f_w_mv2_o],
            'N F': [n_cols],
            'type': ['Full data frame']
            })

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('pastel')
        sns.barplot(x='N F', y='type', data=df_show, color='lightgray', ax=ax,
                    label=f'No missing values ({n_f_wo_mv} • {f_f_wo_mv:.1f}%)')

        sns.set_color_codes('pastel')
        sns.barplot(x='N MV', y='type', data=df_show, color='g', ax=ax,
                    label=f'Not applicable and not available ({n_f_w_mv_1a2} • {f_f_w_mv_1a2:.1f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='N MV 1 xor 2', y='type', data=df_show, color='g', ax=ax,
                    label=f'Not applicable only ({n_f_w_mv1_o} • {f_f_w_mv1_o:.1f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='N MV2 only', y='type', data=df_show, color='g', ax=ax,
                    label=f'Not available only ({n_f_w_mv2_o} • {f_f_w_mv2_o:.1f}%)')

        ax.legend(ncol=1, loc='lower right', frameon=True,
                  title='Type of missing values contained in the feature')
        ax.set(ylabel='', xlabel=f'Number of features (Total {n_cols})')
        sns.despine(left=True, bottom=True, ax=ax)


def plot_feature_wise(indicators, plot=False, show=True, ax=None):
    """Plot the statistics feature-wise."""
    n_mv_fw = indicators['feature-wise']

    n_rows = indicators['global'].at[0, 'n_rows']

    if show:
        with pd.option_context('display.max_rows', None):
            print(
                f'\n'
                f'Statistics feature-wise:\n'
                f'------------------------\n'
                f'\n'
                f'{n_mv_fw}'
            )

    if plot:
        # Plot proportion of missing values in each feature
        # Copy index in a column for the barplot method
        n_mv_fw['feature'] = n_mv_fw.index

        # Add the total number of values for each feature
        n_mv_fw['N V'] = n_rows

        # Get rid of the features with no missing values
        n_mv_fw_l = n_mv_fw[(n_mv_fw['N MV1'] != 0) | (n_mv_fw['N MV2'] != 0)]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        sns.set_color_codes('pastel')
        sns.barplot(x='N V', y='feature', data=n_mv_fw_l, ax=ax,
                    color='b', label=f'Not missing')

        sns.set_color_codes('muted')
        sns.barplot(x='N MV', y='feature', data=n_mv_fw_l, ax=ax,
                    color='b', label=f'Missing - Not applicable')

        sns.set_color_codes("dark")
        sns.barplot(x='N MV2', y='feature', data=n_mv_fw_l, ax=ax,
                    color="b", label=f'Missing - Not available')

        ax.legend(ncol=1, loc='lower right', frameon=True,
                  title='Type of values')
        ax.set(ylabel='', xlabel='Number of values')
        ax.tick_params(labelsize=5)
        sns.despine(left=True, bottom=True, ax=ax)


def plot_rows(indicators, plot=False, show=True, ax=None):
    """Plot stats on rows without missing values."""
    # Get required indicators
    df = pd.concat([indicators['rows'], indicators['global']], axis=1)

    n_r_wo_mv = df.at[0, 'n_r_wo_mv']
    n_r_w_mv = df.at[0, 'n_r_w_mv']
    n_r_w_mv1_o = df.at[0, 'n_r_w_mv1_o']
    n_r_w_mv2_o = df.at[0, 'n_r_w_mv2_o']
    n_r_w_mv_1a2 = df.at[0, 'n_r_w_mv_1a2']
    f_r_wo_mv = df.at[0, 'f_r_wo_mv']
    f_r_w_mv = df.at[0, 'f_r_w_mv']
    f_r_w_mv1_o = df.at[0, 'f_r_w_mv1_o']
    f_r_w_mv2_o = df.at[0, 'f_r_w_mv2_o']
    f_r_w_mv_1a2 = df.at[0, 'f_r_w_mv_1a2']

    n_rows = df.at[0, 'n_rows']

    if show:
        print(
            f'\n'
            f'Statistics on rows:\n'
            f'-------------------\n'
            f'N rows: {n_rows}\n'
            f'N rows without MV:         {n_r_wo_mv} ({f_r_wo_mv:.2f}%)\n'
            f'N rows with MV:            {n_r_w_mv} ({f_r_w_mv:.2f}%)\n'
            f'  N rows with MV1 only:    {n_r_w_mv1_o} ({f_r_w_mv1_o:.2f}%)\n'
            f'  N rows with MV2 only:    {n_r_w_mv2_o} ({f_r_w_mv2_o:.2f}%)\n'
            f'  N rows with MV1 and MV2: {n_r_w_mv_1a2} ({f_r_w_mv_1a2:.2f}%)\n'
        )

    if plot:
        # Plot proportion of features with missing values
        df_show = pd.DataFrame({
            'N MV': [n_r_w_mv],
            'N MV1 only': [n_r_w_mv1_o],
            'N MV2 only': [n_r_w_mv2_o],
            'N MV 1 xor 2': [n_r_w_mv1_o + n_r_w_mv2_o],
            'N R': [n_rows],
            'type': ['Full data frame']
            })

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('pastel')
        sns.barplot(x='N R', y='type', data=df_show, color='lightgray', ax=ax,
                    label=f'No missing values ({n_r_wo_mv} • {f_r_wo_mv:.2f}%)')

        sns.set_color_codes('pastel')
        sns.barplot(x='N MV', y='type', data=df_show, color='r', ax=ax,
                    label=f'Not applicable and not available ({n_r_w_mv_1a2} • {f_r_w_mv_1a2:.2f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='N MV 1 xor 2', y='type', data=df_show, color='r', ax=ax,
                    label=f'Not applicable only ({n_r_w_mv1_o} • {f_r_w_mv1_o:.2f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='N MV2 only', y='type', data=df_show, color='r', ax=ax,
                    label=f'Not available only ({n_r_w_mv2_o} • {f_r_w_mv2_o:.2f}%)')

        ax.legend(ncol=1, loc='lower right', frameon=True,
                  title='Type of missing values contained in the row')
        ax.set(ylabel='', xlabel=f'Number of rows (Total {n_rows})')
        sns.despine(left=True, bottom=True, ax=ax)


def plot_rm_rows(indicators, plot=False, show=True, ax=None):
    """Plot number of rows affected if we remove features with MV."""
    # Get required indicators
    df = pd.concat([indicators['rm_rows'], indicators['global']], axis=1)

    n_r_a_rm_mv1 = df.at[0, 'n_r_a_rm_mv1']
    n_r_a_rm_mv2 = df.at[0, 'n_r_a_rm_mv2']
    n_r_a_rm_mv_1o2 = df.at[0, 'n_r_a_rm_mv_1o2']
    n_r_a_rm_mv1_o = df.at[0, 'n_r_a_rm_mv1_o']
    n_r_a_rm_mv2_o = df.at[0, 'n_r_a_rm_mv2_o']
    n_r_a_rm_mv_1a2 = df.at[0, 'n_r_a_rm_mv_1a2']
    f_r_a_rm_mv1 = df.at[0, 'f_r_a_rm_mv1']
    f_r_a_rm_mv2 = df.at[0, 'f_r_a_rm_mv2']
    f_r_a_rm_mv_1o2 = df.at[0, 'f_r_a_rm_mv_1o2']
    f_r_a_rm_mv1_o = df.at[0, 'f_r_a_rm_mv1_o']
    f_r_a_rm_mv2_o = df.at[0, 'f_r_a_rm_mv2_o']
    f_r_a_rm_mv_1a2 = df.at[0, 'f_r_a_rm_mv_1a2']

    n_rows = df.at[0, 'n_rows']

    if show:
        print(
            f'N rows losing information if we remove features with :\n'
            f'    MV1:          {n_r_a_rm_mv1} ({f_r_a_rm_mv1:.2f}%)\n'
            f'    MV2:          {n_r_a_rm_mv2} ({f_r_a_rm_mv2:.2f}%)\n'
            f'    MV:           {n_r_a_rm_mv_1o2} ({f_r_a_rm_mv_1o2:.2f}%)\n'
            f'    MV1 only:     {n_r_a_rm_mv1_o} ({f_r_a_rm_mv1_o:.2f}%)\n'
            f'    MV2 only:     {n_r_a_rm_mv2_o} ({f_r_a_rm_mv2_o:.2f}%)\n'
            f'    MV1 and MV2:  {n_r_a_rm_mv_1a2} ({f_r_a_rm_mv_1a2:.2f}%)\n'
        )

    if plot:
        # Plot number of rows losing information when removing features with MV
        df_show = pd.DataFrame({
            'N rows affected': [
                n_r_a_rm_mv1,
                n_r_a_rm_mv2,
                n_r_a_rm_mv_1o2,
                n_r_a_rm_mv1_o,
                n_r_a_rm_mv2_o,
                n_r_a_rm_mv_1a2],
            'N R': [n_rows for _ in range(6)],
            'type': [
                f'MV1\n{f_r_a_rm_mv1:.2f}%',
                f'MV2\n{f_r_a_rm_mv2:.2f}%',
                f'MV\n{f_r_a_rm_mv_1o2:.2f}%',
                f'MV1 only\n{f_r_a_rm_mv1_o:.2f}%',
                f'MV2 only\n{f_r_a_rm_mv2_o:.2f}%',
                f'MV1 and MV2\n{f_r_a_rm_mv_1a2:.2f}%']
        })

        df_show.sort_values('N rows affected', ascending=False, inplace=True)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('muted')
        sns.barplot(x='N rows affected', y='type', data=df_show, color='r', ax=ax)

        ax.set_title('Number of rows losing information by'
                     '\nremoving features containing missing values of type:')
        ax.set(ylabel='', xlabel=f'Number of rows (Total {n_rows})')
        sns.despine(left=True, bottom=True, ax=ax)


def plot_rm_features(indicators, plot=False, show=True, ax=None):
    """Plot the part of information lost when removing features with MV."""
    # Get required indicators
    df = pd.concat([indicators['rm_features'], indicators['global']], axis=1)

    n_v_lost_mv1 = df.at[0, 'n_v_lost_mv1']
    n_v_lost_mv2 = df.at[0, 'n_v_lost_mv2']
    n_v_lost_mv_1o2 = df.at[0, 'n_v_lost_mv_1o2']
    n_v_lost_mv1_o = df.at[0, 'n_v_lost_mv1_o']
    n_v_lost_mv2_o = df.at[0, 'n_v_lost_mv2_o']
    n_v_lost_mv_1a2 = df.at[0, 'n_v_lost_mv_1a2']
    f_v_lost_mv1 = df.at[0, 'f_v_lost_mv1']
    f_v_lost_mv2 = df.at[0, 'f_v_lost_mv2']
    f_v_lost_mv_1o2 = df.at[0, 'f_v_lost_mv_1o2']
    f_v_lost_mv1_o = df.at[0, 'f_v_lost_mv1_o']
    f_v_lost_mv2_o = df.at[0, 'f_v_lost_mv2_o']
    f_v_lost_mv_1a2 = df.at[0, 'f_v_lost_mv_1a2']

    n_rows = df.at[0, 'n_rows']
    n_values = df.at[0, 'n_values']

    if show:
        print(
            f'N values lost if we remove features with :\n'
            f'    MV1:          {n_v_lost_mv1} ({f_v_lost_mv1:.2f}%)\n'
            f'    MV2:          {n_v_lost_mv2} ({f_v_lost_mv2:.2f}%)\n'
            f'    MV:           {n_v_lost_mv_1o2} ({f_v_lost_mv_1o2:.2f}%)\n'
            f'    MV1 only:     {n_v_lost_mv1_o} ({f_v_lost_mv1_o:.2f}%)\n'
            f'    MV2 only:     {n_v_lost_mv2_o} ({f_v_lost_mv2_o:.2f}%)\n'
            f'    MV1 and MV2:  {n_v_lost_mv_1a2} ({f_v_lost_mv_1a2:.2f}%)\n'
        )

    if plot:
        # Plot number of values lost when removing features with MV
        df_show = pd.DataFrame({
            'N values lost': [
                n_v_lost_mv1,
                n_v_lost_mv2,
                n_v_lost_mv_1o2,
                n_v_lost_mv1_o,
                n_v_lost_mv2_o,
                n_v_lost_mv_1a2],
            'N R': [n_rows for _ in range(6)],
            'type': [
                f'MV1\n{f_v_lost_mv1:.2f}%',
                f'MV2\n{f_v_lost_mv2:.2f}%',
                f'MV\n{f_v_lost_mv_1o2:.2f}%',
                f'MV1 only\n{f_v_lost_mv1_o:.2f}%',
                f'MV2 only\n{f_v_lost_mv2_o:.2f}%',
                f'MV1 and MV2\n{f_v_lost_mv_1a2:.2f}%']
        })

        df_show.sort_values('N values lost', ascending=False, inplace=True)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('muted')
        sns.barplot(x='N values lost', y='type', data=df_show, color='b', ax=ax)

        ax.set_title('Number of non-missing values lost by'
                     '\nremoving features containing missing values of type:')
        ax.set(ylabel='', xlabel=f'Number of values (Total {n_values})')
        sns.despine(left=True, bottom=True, ax=ax)


def describe_missing_values(df_mv, plot=False):
    """Plot all the indicators."""
    indicators = get_indicators_mv(df_mv)

    _, axes = plt.subplots(3, 1, figsize=(10, 10))

    plot_global(indicators, plot=plot, ax=axes[0])
    plot_features(indicators, plot=plot, ax=axes[1])
    plot_feature_wise(indicators, plot=plot)
    plot_rows(indicators, plot=plot, ax=axes[2])

    _, axes = plt.subplots(2, 1, figsize=(10, 8))

    plot_rm_rows(indicators, plot=plot, ax=axes[0])
    plot_rm_features(indicators, plot=plot, ax=axes[1])
    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser(description='Statistics on missing values.')
parser.add_argument('program')
parser.add_argument('--tag', dest='task_tag', default=None, nargs='?',
                    help='The task tag')
parser.add_argument('--name', dest='db_df_name', default=None, nargs='?',
                    help='The db and df name')
parser.add_argument('--hide', dest='hide', default=False, const=True,
                    nargs='?', help='Whether to plot the stats or print')


def run(argv=None):
    """Show some statistics on the given df."""
    args = parser.parse_args(argv)

    task_tag = args.task_tag
    db_df_name = args.db_df_name
    plot = not args.hide

    if task_tag is not None:
        task = tasks[task_tag]
        db_name, tag = task.meta.db, task.meta.tag
        db = dbs[db_name]
        db.load(task.meta)
        mv = db.missing_values[tag]

    elif db_df_name is not None:
        db_name, df_name = db_df_name.split('/')
        db = dbs[db_name]
        db.load(df_name)
        mv = db.missing_values[df_name]

    else:
        raise ValueError('Incomplete arguments')

    describe_missing_values(mv, plot=plot)
