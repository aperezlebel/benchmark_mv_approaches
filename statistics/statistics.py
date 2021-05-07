"""Compute statistics about missing values on a databse."""
import os
from os.path import join
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from joblib import Memory
from tqdm import tqdm

from prediction.tasks import tasks
from custom.const import get_fig_folder
from .plot_statistics import figure1, figure2, figure2bis, figure3, plot_feature_wise_v2, plot_feature_types
from database import dbs, _load_feature_types
from database.constants import BINARY, CONTINUE_R, CATEGORICAL
from database.constants import is_categorical, is_continue, is_ordinal


memory = Memory('joblib_cache')


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.labelsize': 15,
    'legend.fontsize': 11,
    'figure.figsize': (8, 4.8),
    # 'figure.dpi': 600,
})


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


def every_mv_distribution():
    matplotlib.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 10,
        'axes.labelsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })
    fig, axes = plt.subplots(6, 3, figsize=(6, 9))

    L1 = ['TB/death_pvals', 'TB/hemo', 'TB/hemo_pvals']
    L2 = ['TB/platelet', 'TB/platelet_pvals', 'TB/septic_pvals']
    # L2 = [None, None, None]
    L3 = [None, None, None]
    L4 = [None, None, None]
    L5 = [None, None, None]
    L6 = [None, None, None]
    L3 = ['UKBB/breast_25', 'UKBB/breast_pvals', 'UKBB/fluid_pvals']
    L4 = ['UKBB/parkinson_pvals', 'UKBB/skin_pvals', None]
    L5 = ['MIMIC/hemo_pvals', 'MIMIC/septic_pvals', None]
    L6 = ['NHIS/bmi_pvals', 'NHIS/income_pvals', None]

    L = [L1, L2, L3, L4, L5, L6]

    for i, row in enumerate(tqdm(L)):
        for j, tag in enumerate(row):
            ax = axes[i][j]

            if tag is None:
                ax.axis('off')
                continue

            indicators = cached_indicators(tag, encode_features=False)
            _, _, handles = plot_feature_wise_v2(indicators, ax=ax, plot=True)

            ax.set_title(f'$\\verb|{tag}|$')

    axes[-1, -1].legend(handles, ['Not missing', 'Missing'], fancybox=True, shadow=True, loc='center',)

    return fig, axes


@memory.cache
def cached_indicators(task_tag, encode_features=False):
    task = tasks[task_tag]

    if not encode_features and 'pvals' in task_tag:
        task.meta.encode_select = None
        task.meta.encode_transform = None

    mv = task.mv
    indicators = get_indicators_mv(mv)

    return indicators


def run_mv(args, graphics_folder):
    """Show some statistics on the given df."""
    if args.tag is None:
        every_mv_distribution()

        fig_folder = get_fig_folder(graphics_folder)
        fig_name = 'mv_distribution'

        plt.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
        plt.tight_layout()
        plt.show()

        return

    task_tag = args.tag
    plot = not args.hide

    indicators = cached_indicators(task_tag)

    task = tasks[task_tag]
    db_name = task.meta.db
    df_name = task_tag
    fig1, fig2, fig2b, fig3 = args.fig1, args.fig2, args.fig2b, args.fig3

    if not any((fig1, fig2, fig2b, fig3)):
        fig1, fig2, fig2b, fig3 = True, True, True, True

    # Plot all the indicators
    if fig1:
        figure1(indicators, plot=plot, db_name=db_name, table=df_name)
    if fig2:
        figure2(indicators, plot=plot, db_name=db_name, table=df_name)
    if fig2b:
        figure2bis(indicators, plot=plot, db_name=db_name, table=df_name)
    if fig3:
        figure3(indicators, plot=plot, db_name=db_name, table=df_name)

    fig_folder = get_fig_folder(graphics_folder)

    os.makedirs(join(fig_folder, db_name), exist_ok=True)
    plt.savefig(join(fig_folder, f'{df_name}.pdf'), bbox_inches='tight')
    plt.tight_layout()
    plt.show()


@memory.cache
def cached_prop(task_tag, encode_features=False, T=0):
    task = tasks.get(task_tag, T=T)
    db_name = task.meta.db
    db = dbs[db_name]
    df_name = task.meta.df_name

    # Load types of all inital features of the database
    db_types = _load_feature_types(db, df_name, anonymized=False)

    L = list(task.X.columns)
    L.sort()

    L = [f.split('_')[0] for f in L]
    L = list(set(L))

    if db_name == 'TB':
        task_types = pd.Series(CONTINUE_R, index=L)
    elif db_name == 'UKBB':
        task_types = pd.Series(BINARY, index=L)
    else:
        task_types = pd.Series(CATEGORICAL, index=L)

    # Cast both index to str
    db_types.index = db_types.index.astype(str)
    task_types.index = task_types.index.astype(str)

    task_cols = set(task_types.index)
    db_cols = set(db_types.index)
    missing_cols = task_cols - db_cols
    m = len(missing_cols)
    if m > 0:
        print(f'{m} features not found in DB features:\n{missing_cols}')
    else:
        print('All features found in DB features')
    task_types.update(db_types)

    f_categorical = task_types.map(is_categorical)
    f_ordinal = task_types.map(is_ordinal)
    f_continue = task_types.map(is_continue)

    n_categorical = f_categorical.sum()
    n_ordinal = f_ordinal.sum()
    n_continue = f_continue.sum()

    print(n_categorical, n_ordinal, n_continue)

    assert n_categorical + n_ordinal + n_continue == len(task_types)

    return n_categorical, n_ordinal, n_continue


def run_prop(args, graphics_folder, encode_features=False):
    # task_tag = args.tag


    task_tags = [
        'TB/death_pvals',
        'TB/hemo',
        'TB/hemo_pvals',
        'TB/platelet',
        'TB/platelet_pvals',
        'TB/septic_pvals',
        'UKBB/breast_25',
        'UKBB/breast_pvals',
        'UKBB/fluid_pvals',
        'UKBB/parkinson_pvals',
        'UKBB/skin_pvals',
        'MIMIC/hemo_pvals',
        'MIMIC/septic_pvals',
        'NHIS/bmi_pvals',
        'NHIS/income_pvals',
    ]
    # task_tag = task_tags[0]
    # task = tasks[task_tag]
    # db_name = task.meta.db
    # db = dbs[db_name]
    # df_name = task.meta.df_name

    # # if not encode_features and 'pvals' in task_tag:
    # #     task.meta.encode_select = None
    # #     task.meta.encode_transform = None

    # # Load types of all inital features of the database
    # db_types = _load_feature_types(db, df_name, anonymized=False)
    # # print(db_types)
    # L = list(task.X.columns.astype(str))
    # L.sort()

    # print(f'Features in X:\n{L}')
    # print(len(L))
    # # Kepp parents features only
    # L = [f.split('_')[0] for f in L]
    # L = list(set(L))

    # print(f'Parents features:\n{L}')
    # print(len(L))

    # task_types = pd.Series(CATEGORICAL, index=L)

    # # Cast both index to str
    # db_types.index = db_types.index.astype(str)
    # task_types.index = task_types.index.astype(str)

    # task_cols = set(task_types.index)
    # db_cols = set(db_types.index)
    # missing_cols = task_cols - db_cols
    # m = len(missing_cols)
    # if m > 0:
    #     raise ValueError(f'{m} features not found in DB features:\n{missing_cols}')
    # else:
    #     print('All features found in DB features')
    # task_types.update(db_types)

    # print(task_types)

    # exit()


    rows = []
    for task_tag in task_tags:
        Ts = list(range(5)) if 'pvals' in task_tag else [0]
        print(task_tag)

        db, task = task_tag.split('/')

        for T in Ts:
            n_categorical, n_ordinal, n_continue = cached_prop(task_tag, T=T)
            rows.append([db, task, T, n_categorical, n_ordinal, n_continue])

    props = pd.DataFrame(rows, columns=['db',  'task', 'T', 'categorical', 'ordinal', 'continue'])
    props['tag'] = props['db'].str.cat('/'+props['task'])
    props['n'] = props['categorical'] + props['ordinal'] + props['continue']
    props.set_index(['db', 'task', 'T'], inplace=True)

    print(props)
    plot_feature_types(props)



    # return props




        # fig_folder = get_fig_folder(graphics_folder)
        # fig_name = 'proportion'

        # plt.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
        # plt.tight_layout()
        # plt.show()
