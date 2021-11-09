import os
import re
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from custom.const import get_fig_folder

db_order = [
    'Traumabase',
    'UKBB',
    'MIMIC',
    'NHIS',
]

markers_db = {
    'Traumabase': 'o',
    'UKBB': '^',
    'MIMIC': 'v',
    'NHIS': 's',
}

task_order = [
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

task_order_renamed = [t.replace('_', '\\_').replace('pvals', 'screening') for t in task_order]

rename_db = {
    'TB': 'Traumabase',
}


def run_feature_importance(graphics_folder, results_folder, n, average_folds,
                           mode, hue_by_task):

    def retrive_importance(n):

        filenames = [
            f'{n}_importances.csv',
            f'{n}_mv_props.csv',
        ]

        dfs = []

        # Aggregate feature importances of all tasks
        for root, subdirs, files in os.walk(results_folder):
            print(root)

            res = re.search(join(results_folder, '/(.*)/RS'), root)

            if res is None:
                continue

            if not all([f in files for f in filenames]):
                continue

            task = res.group(1)
            db = task.split('/')[0]
            res = re.search('RS0_T(.)_', root)
            trial = res.group(1)
            task = task.replace('_', '\\_').replace('pvals', 'screening')

            importance = pd.read_csv(join(root, f'{n}_importances.csv'), index_col=0)
            mv_props = pd.read_csv(join(root, f'{n}_mv_props.csv'), index_col=0)
            mv_props.set_index('fold', inplace=True)

            importance.reset_index(inplace=True)
            importance.set_index(['fold', 'repeat'], inplace=True)
            importance_avg = importance.groupby(level='fold').mean()

            if average_folds:
                importance_avg = importance_avg.mean()
                importance_avg = importance_avg.to_frame().T
                mv_props = mv_props.mean()
                mv_props = mv_props.to_frame().T
                id_vars = None
                index = ['feature']

            else:
                importance_avg.reset_index(inplace=True)
                mv_props.reset_index(inplace=True)
                id_vars = ['fold']
                index = ['fold', 'feature']

            importance_avg = pd.melt(importance_avg, id_vars=id_vars,
                                     var_name='feature', value_name='importance_abs')
            importance_avg.set_index(index, inplace=True)

            mv_props = pd.melt(mv_props, id_vars=id_vars, var_name='feature', value_name='mv_prop')
            mv_props.set_index(index, inplace=True)

            df = pd.concat([importance_avg, mv_props], axis=1)
            assert not pd.isna(df).any().any()

            df['db'] = rename_db.get(db, db)
            df['trial'] = trial
            df = pd.concat({task: df}, names=['task'])

            dfs.append(df)

        df = pd.concat(dfs, axis=0)

        df.reset_index(inplace=True)
        df.set_index(['task', 'trial'], inplace=True)
        df_agg = df.groupby(['task', 'trial']).agg({'importance_abs': 'mean'})

        df['importance_ref'] = df_agg
        df['importance_rel'] = df['importance_abs'] - df['importance_ref']
        df['importance_rel_%'] = (df['importance_abs'] - df['importance_ref'])/df['importance_ref']

        return df

    plt.rcParams.update({
        'font.size': 10,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'font.family': 'STIXGeneral',
        'text.usetex': True,
    })

    if n is None:
        sizes = [2500, 10000, 25000, 100000]
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.25, 18))
        legend_bbox = (0.5, 1.32)

    else:
        sizes = [n]
        fig = plt.figure()
        ax = plt.gca()
        axes = [ax]
        legend_bbox = (0.5, 1.28)

    if mode == 'abs':
        y = 'importance_abs'
        yscale = 'symlog'
        linthresh = 0.001
        ylabel = 'Feature importance (score drop)'
    elif mode == 'rel':
        y = 'importance_rel'
        yscale = 'symlog'
        linthresh = 0.001
        ylabel = 'Relative score drop'
    elif mode == 'percent':
        y = 'importance_rel_%'
        yscale = 'symlog'
        linthresh = 1
        ylabel = 'Relative score drop normalized'

    for i, (size, ax) in enumerate(zip(sizes, axes)):
        df = retrive_importance(size)
        print(df)

        sns.set_palette(sns.color_palette('colorblind'))

        if hue_by_task:
            sns.scatterplot(x='mv_prop', y=y, hue='task', style='db', markers=markers_db, data=df,
                            ax=ax, s=15, hue_order=task_order_renamed, linewidth=0.3)
        else:
            sns.scatterplot(x='mv_prop', y=y, hue='db', data=df, ax=ax,
                            s=15, hue_order=db_order, linewidth=0.3)

        ax.set_ylabel(ylabel)
        if i == len(sizes)-1:
            ax.set_xlabel('Proportion of missing values in features')
        else:
            ax.set_xlabel(None)
        ax.set_yscale(yscale, linthresh=linthresh)
        if yscale != 'log':
            ax.axhline(0, xmin=0, xmax=1, color='grey', zorder=-10)#, lw=1)
        ax.set_title(f'n={size}')
        if i == 0:
            ax.legend(title='Database', ncol=4, loc='upper center',
                      bbox_to_anchor=legend_bbox)
        else:
            ax.get_legend().remove()

    fig_name = f'importance_{n}_avg_{mode}' if average_folds else f'importance_{n}_{mode}'
    fig_folder = get_fig_folder(graphics_folder)
    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig.savefig(join(fig_folder, f'{fig_name}.jpg'), bbox_inches='tight', dpi=300)
