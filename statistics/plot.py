from os.path import join
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from prediction.PlotHelper import PlotHelper
from custom.const import get_fig_folder, get_tab_folder

rename = {
    '': 'MIA',
    '_imputed_Mean': 'Mean',
    '_imputed_Mean+mask': 'Mean+mask',
    '_imputed_Med': 'Med',
    '_imputed_Med+mask': 'Med+mask',
    '_imputed_Iterative': 'Iter',
    '_imputed_Iterative+mask': 'Iter+mask',
    '_imputed_KNN': 'KNN',
    '_imputed_KNN+mask': 'KNN+mask',
    '_Logit_imputed_Mean': 'Linear+Mean',
    '_Logit_imputed_Mean+mask': 'Linear+Mean+mask',
    '_Logit_imputed_Med': 'Linear+Med',
    '_Logit_imputed_Med+mask': 'Linear+Med+mask',
    '_Logit_imputed_Iterative': 'Linear+Iter',
    '_Logit_imputed_Iterative+mask': 'Linear+Iter+mask',
    '_Logit_imputed_KNN': 'Linear+KNN',
    '_Logit_imputed_KNN+mask': 'Linear+KNN+mask',
    '_Ridge_imputed_Mean': 'Linear+Mean',
    '_Ridge_imputed_Mean+mask': 'Linear+Mean+mask',
    '_Ridge_imputed_Med': 'Linear+Med',
    '_Ridge_imputed_Med+mask': 'Linear+Med+mask',
    '_Ridge_imputed_Iterative': 'Linear+Iter',
    '_Ridge_imputed_Iterative+mask': 'Linear+Iter+mask',
    '_Ridge_imputed_KNN': 'Linear+KNN',
    '_Ridge_imputed_KNN+mask': 'Linear+KNN+mask',
}

rename_on_plot = {
    'relative_score': 'Relative prediction score',
    'absolute_score': 'Prediction score difference',
    'relative_total_PT': 'Relative total training time',
    'relative_total_WCT': 'Relative wall-clock time',
    'TB': 'Traumabase',
    'Mean+mask': 'Mean\n+mask',
    'Med': 'Median',
    'Med+mask': 'Median\n+mask',
    'Iter': 'Iterative',
    'Iter+mask': 'Iterative\n+mask',
    'KNN+mask': 'KNN\n+mask',
    'Linear+Mean+mask': 'Linear+Mean\n+mask',
    'Linear+Med+mask': 'Linear+Med\n+mask',
    'Linear+Iter+mask': 'Linear+Iter\n+mask',
    'Linear+KNN+mask': 'Linear+KNN\n+mask',
    # 'MIA': 'Boosted trees\n+MIA'
}

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

linear_method_order = [
    # 'MIA', 'Linear+Iter', 'Linear+Iter+mask',
    'MIA',
    'Linear+Mean',
    'Linear+Mean+mask',
    'Linear+Med',
    'Linear+Med+mask',
    'Linear+Iter',
    'Linear+Iter+mask',
    'Linear+KNN',
    'Linear+KNN+mask',
]

db_order = [
    'TB',
    'UKBB',
    'MIMIC',
    'NHIS',
]

tasks_to_drop = {
    'TB': 'platelet',
    'NHIS': 'bmi_pvals',
}


def run_boxplot(graphics_folder, linear):
    # matplotlib.rcParams.update({
    #     # 'font.size': 40,
    #     # 'axes.titlesize': 10,
    #     # 'axes.labelsize': 11,
    #     # 'xtick.labelsize': 8,
    #     # 'ytick.labelsize': 11,
    #     # 'legend.fontsize': 11,
    #     # 'legend.title_fontsize': 12,
    # })
    filepath = 'scores/scores.csv'
    scores = pd.read_csv(filepath, index_col=0)

    # Drop tasks
    scores = scores.set_index(['db', 'task'])
    for db, task in tasks_to_drop.items():
        scores = scores.drop((db, task), axis=0)
    scores = scores.reset_index()

    if linear:
        fig = PlotHelper.plot_MIA_linear(scores, db_order=db_order, method_order=linear_method_order, rename=rename_on_plot)
        xticks = {
            # 0.5: '$\\frac{1}{2}\\times$',
            1/50: '$\\frac{1}{50}\\times$',
            1/10: '$\\frac{1}{10}\\times$',
            1/3: '$\\frac{1}{3}\\times$',
            # 0.75: '$\\frac{3}{4}\\times$',
            1: '$1\\times$',
            # 4/3: '$\\frac{4}{3}\\times$',
            # 10: '$\\frac{10}{1}\\times$',
            3: '$3\\times$',
            10: '$10\\times$',
            # 2: '$2\\times$'
        }
        fig_time = PlotHelper.plot_times(scores, 'PT', xticks_dict=xticks, xlims=(0.005, 15), method_order=linear_method_order, db_order=db_order, rename=rename_on_plot, linear=linear)
    
    else:
        fig = PlotHelper.plot_scores(scores, method_order=method_order, db_order=db_order, rename=rename_on_plot, reference_method=None)
        xticks = {
            # 0.5: '$\\frac{1}{2}\\times$',
            2/3: '$\\frac{2}{3}\\times$',
            # 0.75: '$\\frac{3}{4}\\times$',
            1: '$1\\times$',
            # 4/3: '$\\frac{4}{3}\\times$',
            3/2: '$\\frac{3}{2}\\times$',
            # 2: '$2\\times$'
        }
        fig_time = PlotHelper.plot_times(scores, 'PT', xticks_dict=xticks, method_order=method_order, db_order=db_order, rename=rename_on_plot, linear=linear)

    fig_folder = get_fig_folder(graphics_folder)
    
    fig_name = 'boxplots_scores_linear' if linear else 'boxplots_scores'
    fig_time_name = 'boxplots_times_linear' if linear else 'boxplots_times'

    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig_time.savefig(join(fig_folder, f'{fig_time_name}.pdf'), bbox_inches='tight')
