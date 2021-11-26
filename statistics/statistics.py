"""Compute statistics about missing values on a databse."""
import os
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from custom.const import get_fig_folder, get_tab_folder
from database import _load_feature_types, dbs
from database.constants import (BINARY, CATEGORICAL, CONTINUE_R,
                                is_categorical, is_continue, is_continuous,
                                is_ordinal)
from joblib import Memory
from prediction.tasks import tasks
from tqdm import tqdm

from .plot_statistics import (figure1, figure2, figure2bis, figure3,
                              plot_feature_types, plot_feature_wise_v2)
from .tests import tasks_to_drop

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


task_tags = [
    'TB/death_pvals',
    'TB/hemo',
    'TB/hemo_pvals',
    # 'TB/platelet',
    'TB/platelet_pvals',
    'TB/septic_pvals',
    'UKBB/breast_25',
    'UKBB/breast_pvals',
    'UKBB/fluid_pvals',
    'UKBB/parkinson_pvals',
    'UKBB/skin_pvals',
    'MIMIC/hemo_pvals',
    'MIMIC/septic_pvals',
    # 'NHIS/bmi_pvals',
    'NHIS/income_pvals',
]

db_order = [
    'TB',
    'UKBB',
    'MIMIC',
    'NHIS',
]

db_rename = {
    'TB': 'Traumabase',
}


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
        'axes.labelsize': 8,
        # 'xtick.labelsize': 13,
        # 'ytick.labelsize': 13,
    })
    fig, axes = plt.subplots(6, 3, figsize=(6, 9))

    L1 = ['TB/death_pvals', 'TB/hemo', 'TB/hemo_pvals']
    L2 = ['TB/platelet_pvals', 'TB/septic_pvals', None]
    # L2 = [None, None, None]
    # L3 = [None, None, None]
    # L4 = [None, None, None]
    # L5 = [None, None, None]
    # L6 = [None, None, None]
    L3 = ['UKBB/breast_25', 'UKBB/breast_pvals', 'UKBB/fluid_pvals']
    L4 = ['UKBB/parkinson_pvals', 'UKBB/skin_pvals', None]
    L5 = ['MIMIC/hemo_pvals', 'MIMIC/septic_pvals', None]
    L6 = ['NHIS/income_pvals', None, None]

    L = [L1, L2, L3, L4, L5, L6]

    colors = {
        'TB': 'tab:blue',
        'UKBB': 'tab:orange',
        'MIMIC': 'tab:green',
        'NHIS': 'tab:red',
        # 'TB': 'blue',
        # 'UKBB': 'orange',
        # 'MIMIC': 'green',
        # 'NHIS': 'red',
    }

    handles_dict = {}

    for i, row in enumerate(tqdm(L)):
        for j, tag in enumerate(row):
            ax = axes[i][j]

            if tag is None:
                ax.axis('off')
                continue

            db, task = tag.split('/')
            indicators = cached_indicators(tag, encode_features=False)
            _, _, handles = plot_feature_wise_v2(indicators, ax=ax, plot=True, color=colors[db])

            db = db.replace('TB', 'Traumabase')
            handles_dict[db] = handles[1]

            task = task.replace('_', '\\_')
            task = task.replace('pvals', 'screening')
            ax.set_title(task)

    # axes[-1, -1].legend(handles_dict.values(), handles_dict.keys(),
    #     fancybox=True, shadow=True, loc='center', title='Missing values')
    # p_dummy, = plt.plot([0], marker='None', linestyle='None', label='dummy-tophead')
    # handles_dict[''] = handles[0]
    # handles = [p_dummy]*5+list(handles_dict.values())
    # labels = ['Missing'] + ['']*3 + ['Not missing'] + list(handles_dict.keys())
    # axes[-1, -1].legend(handles, labels, ncol=2,
    # fancybox=True, shadow=True, loc='center',)

    fig.tight_layout()

    db_titles = {
        0: 'Traumabase',
        2: 'UKBB',
        4: 'MIMIC',
        5: 'NHIS',
    }

    db_titles2 = {
        1: 'Traumabase',
        3: 'UKBB',
        4: 'MIMIC',
        5: 'NHIS',
    }

    ax = axes[0]
    fs = 14
    lw = 1.3
    dh = 1./9
    l_tail = 0.03
    pos_arrow = -0.45

    for i, db in db_titles2.items():
        # Here is the label and arrow code of interest
        # axes[i, 0].annotate(db, xy=(-0.4, 0.5), xycoords='axes fraction',
        #             fontsize=fs, ha='center', va='center',
        #             bbox=None,#dict(boxstyle='square', fc='white'),
        #             # arrowprops=dict(arrowstyle=f'-[, widthB={70/fs}, lengthB=0.5', lw=lw),
        #             rotation=90,
        #             )
        axes[i, -1].annotate(db, xy=(0.5, 0.5), xycoords='axes fraction',
                    fontsize=fs, ha='center', va='center',
                    bbox=dict(boxstyle='square', fc='white'),
                    # arrowprops=dict(arrowstyle=f'-[, widthB={70/fs}, lengthB=0.5', lw=lw),
                    rotation=0,
                    )

    # x, y = np.array([[0, 1], ])
    dh = 1./6
    # for i in range(0,7):
    for i in [1.05, 2.02, 3.96]:
        y = i*dh
        line = matplotlib.lines.Line2D([0, 1], [y, y], lw=0.5, ls='-', color='black',
            alpha=1, transform=fig.transFigure)
    # axes[1, 0].add_line(line)
        fig.add_artist(line)

    axes[-1, 0].set_xlabel('Features')
    for ax in axes[:, 0]:
        ax.set_ylabel('Proportion')
    # axes[-1, 0].set_ylabel('Proportion')

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
def cached_types(task_tag, encode_features=False, T=0):
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

    return task_types


def get_prop(task_tag, encode_features=False, T=0):
    task_types = cached_types(task_tag, encode_features, T=T)

    f_categorical = task_types.map(is_categorical)
    f_ordinal = task_types.map(is_ordinal)
    f_continuous = task_types.map(is_continuous)

    n_categorical = f_categorical.sum()
    n_ordinal = f_ordinal.sum()
    n_continuous = f_continuous.sum()

    print(n_categorical, n_ordinal, n_continuous)

    assert n_categorical + n_ordinal + n_continuous == len(task_types)

    return n_categorical, n_ordinal, n_continuous


def run_prop(args, graphics_folder):

    rows = []
    for task_tag in task_tags:
        Ts = list(range(5)) if 'pvals' in task_tag else [0]
        print(task_tag)

        db, task = task_tag.split('/')

        for T in Ts:
            n_categorical, n_ordinal, n_continuous = get_prop(task_tag, T=T)
            rows.append([db_rename.get(db, db), task, T, n_categorical, n_ordinal, n_continuous])

    props = pd.DataFrame(rows, columns=['db',  'task', 'T', 'categorical', 'ordinal', 'continuous'])
    props['tag'] = props['db'].str.cat('/'+props['task'])
    props['n'] = props['categorical'] + props['ordinal'] + props['continuous']

    # Drop tasks
    props = props.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        props = props.drop((db, task), axis=0, errors='ignore')
    props = props.reset_index()

    props.set_index(['db', 'task', 'T'], inplace=True)
    print(props)

    plot_feature_types(props)

    fig_folder = get_fig_folder(graphics_folder)
    fig_name = 'proportion'

    plt.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.tight_layout(pad=0.3)
    # plt.show()


def compute_correlation(_X):
    """Compute the pairwise correlation from observations of a feature vector.

    Similar to numpy.corrcoef except that it ignores missing observations in X.

    Parameters
    ----------
    X : np.array of shape (k, n)
        Matrix containing the n observations of the k features

    Returns
    -------
    R : np.array of shape (k, k)
        Pairwise correlation coefficients
    N : np.array of shape (k, k)
        Number of values taken for correlation computation of pair of features
        Number of values taken for correlation computation of pair of features

    """
    X = np.array(_X)
    k, n = X.shape
    R = np.nan*np.ones((k, k))
    N = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            x1 = X[i, :]
            x2 = X[j, :]

            # Select index for which no missing values are in x1 nor x2
            idx = ~np.isnan(x1) & ~np.isnan(x2)
            n_values = np.sum(idx)
            if n_values < 100:
                print(f'Warning: only {n_values} values taken for correlation.')

            R[i, j] = np.nan if n_values < 3 else np.corrcoef([x1[idx], x2[idx]])[0, 1]
            N[i, j] = n_values

    if isinstance(_X, pd.DataFrame):
        features = _X.index
        R = pd.DataFrame(R, index=features, columns=features)

    return R, N


@memory.cache
def cached_task_correlation(task_tag, encode_features=False, T=0):
    task = tasks.get(task_tag, T=T)
    db_name = task.meta.db
    db = dbs[db_name]
    df_name = task.meta.df_name

    task_types = cached_types(task_tag, encode_features=encode_features, T=T)


    f_categorical = task_types.map(is_categorical)
    f_ordinal = task_types.map(is_ordinal)
    f_continue = task_types.map(is_continue)


    f_selected = f_ordinal | f_continue
    f_selected = f_selected[f_selected]

    X_selected = task.X[f_selected[f_selected].index]

    R, N = compute_correlation(X_selected.T)


    return R, N


def run_cor(args, graphics_folder, absolute=False, csv=False, prop_only=True):
    thresholds = [0.1, 0.2, 0.3]

    rows = []
    for task_tag in task_tags:
        Ts = list(range(5)) if 'pvals' in task_tag else [0]
        print(task_tag)

        db, task = task_tag.split('/')

        for T in Ts:
            R, _ = cached_task_correlation(task_tag, T=T)
            for threshold in thresholds:
                if absolute:
                    R = R.abs()
                N = (R > threshold).sum(axis=1)
                N_mean = N.mean()
                k = R.shape[0]
                rows.append([db_rename.get(db, db), task, T, threshold, k, N_mean, N_mean/k])

    df_cor = pd.DataFrame(rows, columns=['db',  'task', 'T', 'threshold', 'n_selected', 'N_mean', 'prop'])
    df_n_selected = df_cor.groupby(['db',  'task']).agg({'n_selected': 'mean'})
    df_cor = df_cor.pivot(index=['db', 'task', 'T'], columns='threshold', values=['N_mean', 'prop'])
    df_cor = df_cor.groupby(['db',  'task'])
    df_cor = df_cor.agg('mean')

    df_cor_mean = df_cor.mean()
    df_cor_mean = pd.DataFrame(df_cor_mean).T
    df_cor_mean.index = pd.MultiIndex.from_tuples([('Average', '')])
    df_cor = pd.concat([df_cor, df_cor_mean], axis=0)
    df_n_selected.loc[('Average', ''), 'n_selected'] = float(df_n_selected.mean())

    def to_int(x):  # Convert to int and robust to NaN
        try:
            return str(int(x))
        except:
            return x

    def to_percent(x):  # Convert to int and robust to NaN
        try:
            return f'{int(100*x)}\\%'
        except:
            return x

    df_cor['N_mean'] = df_cor['N_mean'].applymap(to_int)
    df_cor['prop'] = df_cor['prop'].applymap(to_percent)
    df_cor['n_selected'] = df_n_selected.applymap(to_int)
    df_cor.set_index('n_selected', append=True, inplace=True)

    db_order_renamed = [db_rename.get(db, db) for db in db_order]
    df_cor = df_cor.reindex(db_order_renamed+['Average'], level=0, axis=0)

    # Rotate Traumabase
    df_cor.rename({'Traumabase': '\\rotsmash{Traumabase}'}, axis=0, inplace=True)

    # Processing for dumping
    df_cor.index.rename(['Database', 'Task', '\\# features'], inplace=True)
    df_cor.index = df_cor.index.set_levels(df_cor.index.levels[1].str.replace('pvals', 'screening'), level=1)
    df_cor.index = df_cor.index.set_levels(df_cor.index.levels[1].str.replace('_', r'\_'), level=1)

    if prop_only:
        df_cor.drop(['N_mean'], axis=1, inplace=True)
        df_cor.rename({'prop': 'Threshold'}, axis=1, inplace=True)
        df_cor.columns.rename(['', ''], inplace=True)

    else:
        df_cor.columns.rename(['', 'Threshold'], inplace=True)
        df_cor.rename({'N_mean': r'$\bar{n}$', 'prop': r'$\bar{p}$'}, axis=1, inplace=True)

    smallskip = '0.15in'
    bigskip = '0.2in'
    index_rename = {}
    for i, v in enumerate(pd.unique(df_cor.index.get_level_values(0))):
        if i == 0:
            continue
        skip = bigskip if v == 'Average' else smallskip
        index_rename[v] = f'\\rule{{0pt}}{{{skip}}}{v}'

    df_cor.rename(index_rename, axis=0, level=0, inplace=True)

    tab_folder = get_tab_folder(graphics_folder)
    tab_name = 'correlation_abs' if absolute else 'correlation'

    print(df_cor)

    df_cor.to_latex(join(tab_folder, f'{tab_name}.tex'), na_rep='', escape=False)

    if csv:
        df_cor.to_csv(join(tab_folder, f'{tab_name}.csv'))


def run_time():
    # path = os.path.abspath('scores/scores.csv')
    # df = pd.read_csv(path, index_col=0)
    filepaths = [
        'scores/scores.csv',
        'scores/scores_mi_2500.csv',
        'scores/scores_mia_2500.csv',
        'scores/scores_mi_10000.csv',
        'scores/scores_mia_10000.csv',
        'scores/scores_mia_25000.csv',
        'scores/scores_mi_25000.csv',
        'scores/scores_mia_100000.csv',
        'scores/scores_mi_100000.csv',
        'scores/scores_mean+mask+bagging_2500.csv',
        'scores/scores_mean+mask+bagging_10000.csv',
        'scores/scores_mean+mask+bagging_25000.csv',
    ]
    dfs = [pd.read_csv(path, index_col=0) for path in filepaths]
    df = pd.concat(dfs, axis=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        df.drop(index=df[(df['db'] == db) & (df['task'] == task)].index, inplace=True)

    df['task'] = df['task'].str.replace('_pvals', '_screening')

    # Sum times
    df['total_PT'] = df['imputation_PT'].fillna(0) + df['tuning_PT']
    df['total_WCT'] = df['imputation_WCT'].fillna(0) + df['tuning_WCT']

    total_pt = df['total_PT'].sum()
    total_wct = df['total_WCT'].sum()

    print(
        f'Total training time:\n'
        f'\tCPU time: {int(total_pt/3600)} hours\n'
        f'\tWall-clock time: {int(total_wct/3600)} hours'
    )
    method = 'MIA'

    df = df.query('method == @method')

    # Sum times
    df['total_PT'] = df['imputation_PT'].fillna(0) + df['tuning_PT']
    df['total_WCT'] = df['imputation_WCT'].fillna(0) + df['tuning_WCT']

    total_pt = df['total_PT'].sum()
    total_wct = df['total_WCT'].sum()

    print(
        f'{method} training time:\n'
        f'\tCPU time: {int(total_pt/3600)} hours\n'
        f'\tWall-clock time: {int(total_wct/3600)} hours'
    )


# 150 000 heures CPU, noeuds de 40 CPU, 3750 heures noeuds, 156 jours noeuds, 20 noeuds : 8 jours

# iter : 1486 CPU hours
# + mask: 1836 CPU hours

# Les deux : 3322 CPU hours
# 332 200 heures CPU
# noeuds de 40 CPU
# 8305 heures noeuds
# 350 jours
# 40 noeuds : 9 jours


# MIA : 950 hours CPU
# 95 000 hours

