import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import itertools

from prediction.df_utils import get_ranks_tab, aggregate
from custom.const import get_fig_folder, get_tab_folder


tasks_to_drop = {
    'TB': 'platelet',
    'NHIS': 'bmi_pvals',
}


def run_difficulty(graphics_folder, averaged_scores=True):
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
    ranks = get_ranks_tab(scores, method_order=method_order, db_order=db_order,
                          average_sizes=False, average_on_dbs=False)

    melted_ranks = pd.melt(ranks, ignore_index=False, value_name='Rank')
    melted_ranks.reset_index(inplace=True)
    melted_ranks.set_index(['Size', 'Database', 'Task', 'Method'], inplace=True)

    scores = aggregate(scores, 'score')
    scores.rename({
        'size': 'Size',
        'db': 'Database',
        'task': 'Task',
        'method': 'Method',
    }, inplace=True, axis=1)
    scores.set_index(['Size', 'Database', 'Task', 'Method'], inplace=True)

    scores = scores[['score', 'scorer']]
    melted_ranks = melted_ranks.astype(float)
    melted_ranks['Score'] = scores['score']
    melted_ranks['scorer'] = scores['scorer']
    melted_ranks.dropna(axis=0, inplace=True)

    scores = melted_ranks

    if averaged_scores:
        for _, group in scores.groupby(['Size', 'Database', 'Task']):
            group['Score'] = group['Score'].mean()
            scores.update(group['Score'])

    scores.reset_index(inplace=True)

    scores_auc = scores.query('scorer == "roc_auc_score"')
    scores_r2 = scores.query('scorer == "r2_score"')

    fig1 = plt.figure(figsize=(5, 3.3))
    ax = plt.gca()

    # Build the color palette
    paired_colors = sns.color_palette('Paired').as_hex()
    boxplot_palette = sns.color_palette(['#525252']+paired_colors)
    sns.set_palette(boxplot_palette)

    # AUC figure
    sns.scatterplot(x='Score', y='Rank', hue='Method', data=scores_auc,
                    ax=ax, hue_order=method_order, s=20)
    palette = itertools.cycle(sns.color_palette())

    for _, group in scores_auc.groupby('Method', sort=False):
        z = sm.nonparametric.lowess(group['Rank'], group['Score'])
        ax.plot(z[:, 0], z[:, 1], color=next(palette))

    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel('AUC score')
    ax.invert_yaxis()
    
    # Rename methods in legend
    handles, labels = ax.get_legend_handles_labels()
    renamed_labels = [rename.get(label, label) for label in labels]
    ax.legend(title='Method', handles=handles, labels=renamed_labels, bbox_to_anchor=(1, 1))
    
    plt.tight_layout()

    # R2 figure
    fig2 = plt.figure(figsize=(5, 3.3))
    ax = plt.gca()

    sns.scatterplot(x='Score', y='Rank', hue='Method', data=scores_r2,
                    ax=ax, hue_order=method_order, s=20)
    palette = itertools.cycle(sns.color_palette())

    for _, group in scores_r2.groupby('Method', sort=False):
        z = sm.nonparametric.lowess(group['Rank'], group['Score'])
        ax.plot(z[:, 0], z[:, 1], color=next(palette))

    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel('$r^2$ score')
    ax.invert_yaxis()

    # Rename methods in legend
    handles, labels = ax.get_legend_handles_labels()
    renamed_labels = [rename.get(label, label) for label in labels]
    ax.legend(title='Method', handles=handles, labels=renamed_labels, bbox_to_anchor=(1, 1))
    
    plt.tight_layout()

    fig_folder = get_fig_folder(graphics_folder)
    fig_name = 'rank_vs_difficulty'

    fig1.savefig(os.path.join(fig_folder, f'{fig_name}_auc.pdf'), bbox_inches='tight', pad_inches=0)
    fig1.savefig(os.path.join(fig_folder, f'{fig_name}_auc.jpg'), bbox_inches='tight', pad_inches=0)

    fig2.savefig(os.path.join(fig_folder, f'{fig_name}_r2.pdf'), bbox_inches='tight', pad_inches=0)
    fig2.savefig(os.path.join(fig_folder, f'{fig_name}_r2.jpg'), bbox_inches='tight', pad_inches=0)
