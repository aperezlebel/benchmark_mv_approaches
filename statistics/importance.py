import copy
import os
import re
from functools import reduce
from os.path import join

import matplotlib
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

    colors = [
        sns.color_palette('Set2', n_colors=8).as_hex(),
        sns.color_palette('husl', n_colors=5).as_hex(),
    ]
    colors = reduce(lambda x, y: x+y, colors)

    for i, (size, ax) in enumerate(zip(sizes, axes)):
        df = retrive_importance(size)
        print(df)

        if hue_by_task:
            sns.set_palette(sns.color_palette(colors))
            sns.scatterplot(x='mv_prop', y=y, hue='task', style='db', markers=markers_db, data=df,
                            ax=ax, s=15, hue_order=task_order_renamed, style_order=db_order, linewidth=0.3)#, legend=False)
            ncol = 3
            legend_bbox = (0.5, 1.7)
            title = 'Task'
        else:
            sns.set_palette(sns.color_palette('colorblind'))
            sns.scatterplot(x='mv_prop', y=y, hue='db', data=df, ax=ax,
                            s=15, hue_order=db_order, linewidth=0.3)
            ncol = 4
            title = 'Database'

        ax.set_ylabel(ylabel)
        if i == len(sizes)-1:
            ax.set_xlabel('Proportion of missing values in features')
        else:
            ax.set_xlabel(None)

        ax.set_yscale(yscale, linthresh=linthresh)
        if yscale != 'log':
            ax.axhline(0, xmin=0, xmax=1, color='grey', zorder=-10)#, lw=1)

        ax.set_title(f'n={size}')

        # Update legend
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            if hue_by_task:
                task_handles = handles[1:14]
                db_markers = ax.collections[-4:]

                # Update markers shapes in the legend (all had same shape)
                for i in range(13):
                    if i < 5:
                        id_marker = 0
                    elif i < 10:
                        id_marker = 1
                    elif i < 12:
                        id_marker = 2
                    else:
                        id_marker = 3
                    h = task_handles[i]
                    fc = h.get_fc()
                    ec = h.get_ec()
                    task_handles[i] = copy.copy(db_markers[id_marker])
                    task_handles[i].set_fc(fc)
                    task_handles[i].set_ec(ec)

                task_labels = labels[1:14]
                task_labels = [t.split('/')[1] for t in task_labels]
                blank_handle = [handles[0]]
                blank_label = ['']
                # 3 columns legend
                handles = (
                    blank_handle
                    + task_handles[:5]
                    + blank_handle
                    + task_handles[5:10]
                    + blank_handle
                    + task_handles[10:12]
                    + 2*blank_handle
                    + task_handles[12:13])
                labels = (
                    ['Traumabase']
                    + task_labels[:5]
                    + ['UKBB']
                    + task_labels[5:10]
                    + ['MIMIC']
                    + task_labels[10:12]
                    + blank_label
                    + ['NHIS']
                    + task_labels[12:13])
                # 4 columns legend
                # handles = (blank_handle
                # + task_handles[:5]
                # + blank_handle
                # + task_handles[5:10]
                # + blank_handle
                # + task_handles[10:12]
                # + 4*blank_handle
                # + task_handles[12:13]
                # + 4*blank_handle)
                # labels = (['Traumabase']
                # + task_labels[:5]
                # + ['UKBB']
                # + task_labels[5:10]
                # + ['MIMIC']
                # + task_labels[10:12]
                # + 3*blank_label
                # + ['NHIS']
                # + task_labels[12:13]
                # + 4*blank_label)

            ax.legend(title=title, ncol=ncol, loc='upper center',
                      bbox_to_anchor=legend_bbox,
                      handles=handles, labels=labels,
                      )

        else:
            ax.get_legend().remove()

    fig_name = f'importance_{n}_avg_{mode}_hue{hue_by_task}' if average_folds else f'importance_{n}_{mode}'
    fig_folder = get_fig_folder(graphics_folder)
    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig.savefig(join(fig_folder, f'{fig_name}.jpg'), bbox_inches='tight', dpi=300)
