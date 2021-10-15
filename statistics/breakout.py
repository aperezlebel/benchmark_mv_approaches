import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from custom.const import get_fig_folder
from prediction.df_utils import aggregate

tasks_to_drop = {
    'TB': 'platelet',
    'NHIS': 'bmi_pvals',
}


def run_breakout(graphics_folder, linear):
    filepath = 'scores/scores.csv'
    scores = pd.read_csv(filepath, index_col=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        scores.drop(index=scores[(scores['db'] == db) & (scores['task'] == task)].index, inplace=True)

    scores['task'] = scores['task'].str.replace('_pvals', '_screening')

    method_order = [
        'MIA',
        'Mean',
        'Mean+mask',
        'Med',
        'Med+mask',
        'Iter',
        'Iter+mask',
        'KNN',
        'KNN+mask',
    ]

    db_order = [
        'TB',
        'UKBB',
        'MIMIC',
        'NHIS',
    ]

    rename = {
        'Med': 'Median',
        'Med+mask': 'Median+mask',
        'Iter': 'Iterative',
        'Iter+mask': 'Iterative+mask',
    }

    scores = scores.query('method in @method_order')
    scores = aggregate(scores, 'score')
    scores.rename({
        'size': 'Size',
        'db': 'Database',
        'task': 'Task',
        'method': 'Method',
    }, inplace=True, axis=1)
    scores.set_index(['Size', 'Database', 'Task', 'Method'], inplace=True)
    scores = scores.reindex(method_order, level=3)
    scores.reset_index(inplace=True)
    # scores.set_index(['Database', 'Task'], inplace=True)

    # print(scores)

    # Build the color palette
    paired_colors = sns.color_palette('Paired').as_hex()
    boxplot_palette = sns.color_palette(['#525252']+paired_colors)
    sns.set_palette(boxplot_palette)

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

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    plt.rcParams.update({
        'text.usetex': True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        # 'axes.labelsize': 15,
        'legend.fontsize': 9,#10,
        # 'figure.figsize': (8, 4.8),
        # 'figure.dpi': 600,
    })
    fig, axes = plt.subplots(6, 3, figsize=(8, 13))

    # plt.figure()
    # ax = plt.gca()

    positions = {
        'TB/death_screening': (0, 0),
        'TB/hemo': (0, 1),
        'TB/hemo_screening': (0, 2),
        'TB/platelet_screening': (1, 0),
        'TB/septic_screening': (1, 1),
        'UKBB/breast_25': (2, 0),
        'UKBB/breast_screening': (2, 1),
        'UKBB/fluid_screening': (2, 2),
        'UKBB/parkinson_screening': (3, 0),
        'UKBB/skin_screening': (3, 1),
        'MIMIC/hemo_screening': (4, 0),
        'MIMIC/septic_screening': (4, 1),
        'NHIS/income_screening': (5, 0),
    }

    for i in range(6):
        for j in range(3):
            axes[i, j].axis('off')

    for name, group in scores.groupby(['Database', 'Task']):
        # print(group)
        db = group['Database'].iloc[0]
        task = group['Task'].iloc[0]
        i, j = positions.get(f'{db}/{task}', None)
        ax = axes[i, j]
        ax.axis('on')
        # print(group)
        # print(group)
        # group['score'] = pd.to_numeric(group['score'])
        # group = group.astype({'score': float})#, 'Size': str})
        group = group.astype({'Size': str})
        # group['const'] = '0'
        # sns.stripplot(x='score', hue='Method', data=group)
        # sns.swarmplot(x='score', hue='Method', data=group)
        # sns.scatterplot(x='score', y=1, hue='Method', data=group)
        # sns.swarmplot(x='score', y='Size', hue='Method', data=group)
        # sns.stripplot(x='score', y='const', hue='Method', data=group)

        # Seed for jitter
        np.random.seed(0)
        # sns.stripplot(x='score', y='Size', hue='Method', data=group, ax=ax,
        sns.stripplot(x='Size', y='score', hue='Method', data=group, ax=ax,
                    #   order=['100000', '25000', '10000', '2500'], s=4, jitter=1)#0.3)
                      order=['2500', '10000', '25000', '100000'], s=4, jitter=1)#0.3)
        xlabels = {
            'roc_auc_score': 'AUC',
            'r2_score': '$r^2$',
        }
        # ax.set_xlabel()
        ax.set_ylabel('Score')
        # ax.set_title(f"\\verb|{group['Task'].iloc[0]}|")
        ax.set_title(group['Task'].iloc[0].replace('_', '\\_'))
        # sns.swarmplot(x='score', y='Size', hue='Method', data=group)
        if (i, j) == (5, 0):
            # Rename methods in legend
            handles, labels = ax.get_legend_handles_labels()
            renamed_labels = [rename.get(label, label) for label in labels]
            ax.legend(title='Method', ncol=2, bbox_to_anchor=(1.05, 1.0),
                      handles=handles, labels=renamed_labels)
        else:
            ax.get_legend().remove()
        if j >= 1:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel(None)
        if i < 5:
            ax.set_xlabel(None)

        scorer = xlabels[group['scorer'].iloc[0]]
        # ax.annotate(scorer, xy=(0.018, 0.028), xycoords='axes fraction',
        #             bbox=dict(boxstyle='square', ec='black', fc='white', alpha=1, linewidth=0.7),
        #             ha='left', va='bottom', fontsize=8)

        ax.annotate(scorer, xy=(0.019, 0.972), xycoords='axes fraction',
                    bbox=dict(boxstyle='square', ec='black', fc='white', alpha=1, linewidth=0.7),
                    ha='left', va='top', fontsize=8)

        # ax.annotate(scorer, xy=(0.981, 0.028), xycoords='axes fraction',
        #             bbox=dict(boxstyle='square', ec='black', fc='white', alpha=1, linewidth=0.7),
        #             ha='right', va='bottom', fontsize=8)


        # ax.tick_params(axis="x",direction="in", pad=-15)
        # break

    db_titles2 = {
        1: 'Traumabase',
        3: 'UKBB',
        4: 'MIMIC',
        5: 'NHIS',
    }

    fs = 14

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
    dh = 1./6
    # for i in range(0,7):
    for i in [1.35, 2.15, 3.77]:
        y = i*dh
        line = matplotlib.lines.Line2D([0.05, 0.9], [y, y], lw=0.5, ls='-', color='gray',
            alpha=1, transform=fig.transFigure)
    # axes[1, 0].add_line(line)
        fig.add_artist(line)

    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    fig_folder = get_fig_folder(graphics_folder)
    fig_name = 'breakout'

    plt.savefig(os.path.join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(fig_folder, f'{fig_name}.jpg'), bbox_inches='tight', dpi=400)
    plt.tight_layout()
    # plt.show()
