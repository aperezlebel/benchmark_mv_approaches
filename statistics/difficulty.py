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


def run_difficulty(graphics_folder):
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

    # print(scores)

    scores = scores.query('method in @method_order')

    # print(scores)

    # exit()

    ranks = get_ranks_tab(scores, method_order=method_order, db_order=db_order,
                          average_sizes=False, average_on_dbs=False)
    # sizes = ranks.index.get_level_values(0).unique()
    # print(scores)
    # print(ranks)

    # ranks = ranks.reset_index()

    melted_ranks = pd.melt(ranks, ignore_index=False, value_name='Rank')
    
    # melted_ranks = melted_ranks.reset_index()

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

    # print(melted_ranks)
    # print(scores)

    melted_ranks = melted_ranks.astype(float)

    # print(melted_ranks)
    # print(scores)
    # exit()

    # scores['Rank'] = melted_ranks['Rank']
    # print(scores)
    # exit()
    melted_ranks['Score'] = scores['score']
    melted_ranks['scorer'] = scores['scorer']

    print(melted_ranks)

    melted_ranks.dropna(axis=0, inplace=True)

    print(melted_ranks)

    # exit()

    print(scores)

    scores = melted_ranks

    # exit()
    # print(list(scores.columns))

    # print(scores['scorer'])

    scores.reset_index(inplace=True)

    # scores = scores.query('Method == "MIA"')

    scores_auc = scores.query('scorer == "roc_auc_score"')
    scores_r2 = scores.query('scorer == "r2_score"')

    print(scores_auc)
    print(scores_r2)

    fig1 = plt.figure()
    ax = plt.gca()

    # Build the color palette for the boxplot
    paired_colors = sns.color_palette('Paired').as_hex()
    boxplot_palette = sns.color_palette(['#525252']+paired_colors)

    # Boxplot
    sns.set_palette(boxplot_palette)

    sns.scatterplot(x='Score', y='Rank', hue='Method', data=scores_auc,
                    ax=ax, hue_order=method_order)
    palette = itertools.cycle(sns.color_palette())
    for name, group in scores_auc.groupby('Method', sort=False):
        print(name)
        z = sm.nonparametric.lowess(group['Rank'], group['Score'])
        # print(z)
        ax.plot(z[:, 0], z[:, 1], color=next(palette))
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel('AUC score')
    # sns.lineplot()
    # twiny = ax.twiny()

    plt.tight_layout()

    fig2 = plt.figure()
    ax = plt.gca()

    sns.scatterplot(x='Score', y='Rank', hue='Method', data=scores_r2,
                    ax=ax, hue_order=method_order)
    palette = itertools.cycle(sns.color_palette())
    for name, group in scores_r2.groupby('Method', sort=False):
        print(name)
        z = sm.nonparametric.lowess(group['Rank'], group['Score'])
        # print(z)
        ax.plot(z[:, 0], z[:, 1], color=next(palette))
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_xlabel('$r^2$ score')

    # sns.scatterplot(x='score', y='Rank', hue='Method', data=scores_r2, ax=twiny, marker='x')
    # for name, group in scores_r2.groupby('Method'):
    #     z = sm.nonparametric.lowess(group['Rank'], group['score'])
    #     # print(z)
    #     twiny.plot(z[:, 0], z[:, 1], linestyle=':')
    # # sns.lineplot()
    
    plt.tight_layout()

    fig_folder = get_fig_folder(graphics_folder)
    fig_name = 'rank_vs_difficulty'

    fig1.savefig(os.path.join(fig_folder, f'{fig_name}_auc.pdf'), bbox_inches='tight', pad_inches=0)
    fig1.savefig(os.path.join(fig_folder, f'{fig_name}_auc.jpg'), bbox_inches='tight', pad_inches=0)

    fig2.savefig(os.path.join(fig_folder, f'{fig_name}_r2.pdf'), bbox_inches='tight', pad_inches=0)
    fig2.savefig(os.path.join(fig_folder, f'{fig_name}_r2.jpg'), bbox_inches='tight', pad_inches=0)

    plt.show() 
