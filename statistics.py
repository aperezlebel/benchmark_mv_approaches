"""Compute statistics about missing values on a databse."""

import argparse
import pandas as pd
import numpy as np

from prediction.tasks import tasks
from plot_statistics import plot_global


def get_indicators_mv(df_mv):
    """Compute indicators about missing values. Used for plotting figures."""
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
    n_mv_fw['N V'] = n_rows
    n_mv_fw['N NMV'] = n_mv_fw['N V'] - n_mv_fw['N MV']
    n_mv_fw['F MV1'] = 100*n_mv_fw['N MV1']/n_rows
    n_mv_fw['F MV2'] = 100*n_mv_fw['N MV2']/n_rows
    n_mv_fw['F MV'] = 100*n_mv_fw['N MV']/n_rows
    n_mv_fw['id'] = np.arange(0, n_mv_fw.shape[0])

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


def run(argv=None):
    """Show some statistics on the given df."""
    parser = argparse.ArgumentParser(description='Stats on missing values.')
    parser.add_argument('program')
    parser.add_argument('--tag', dest='task_tag', default=None, nargs='?',
                        help='The task tag')
    parser.add_argument('--hide', dest='hide', default=False, const=True,
                        nargs='?', help='Whether to plot the stats or print')
    args = parser.parse_args(argv)

    task_tag = args.task_tag
    plot = not args.hide

    task = tasks[task_tag]
    mv = task.mv
    indicators = get_indicators_mv(mv)
    print(indicators)
    plot_global(indicators, plot=plot)
