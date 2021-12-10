from os.path import join
import pandas as pd

from custom.const import get_fig_folder
from prediction.PlotHelper import PlotHelper
from .tests import run_wilcoxon

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
    filepath = 'scores/scores.csv'
    scores = pd.read_csv(filepath, index_col=0)

    # Drop tasks
    for db, task in tasks_to_drop.items():
        scores.drop(index=scores[(scores['db'] == db) & (scores['task'] == task)].index, inplace=True)

    scores['task'] = scores['task'].str.replace('_pvals', '_screening')

    # Get Wilcoxon table for symbol annotation
    W_test = run_wilcoxon(graphics_folder=None, spacing=False, no_rename=True)

    symbols = {}

    def pvalue_to_symbol(pvalue, alpha, n_bonferroni):
        if pvalue < alpha/n_bonferroni:
            return '$\\star\\star$'
        if pvalue < alpha:
            return '$\\star$'
        return None

    alpha = 0.05
    n_bonferroni = W_test.shape[0]

    for size in W_test:
        symbols[size] = {k: pvalue_to_symbol(v, alpha, n_bonferroni) for k, v in W_test[size].iteritems()}
        symbols[size]['MIA'] = '$\\rightarrow$'

    if linear:
        fig = PlotHelper.plot_MIA_linear(
            scores, db_order=db_order, method_order=linear_method_order,
            rename=rename_on_plot, symbols=symbols)
        xticks = {
            1/50: '$\\frac{1}{50}\\times$',
            1/10: '$\\frac{1}{10}\\times$',
            1/3: '$\\frac{1}{3}\\times$',
            1: '$1\\times$',
            3: '$3\\times$',
            10: '$10\\times$',
        }
        fig_time = PlotHelper.plot_times(
            scores, 'PT', xticks_dict=xticks, xlims=(0.005, 15),
            method_order=linear_method_order, db_order=db_order,
            rename=rename_on_plot, linear=linear)

    else:
        fig = PlotHelper.plot_scores(
            scores, method_order=method_order, db_order=db_order,
            rename=rename_on_plot, reference_method=None, symbols=symbols)
        xticks = {
            2/3: '$\\frac{2}{3}\\times$',
            1: '$1\\times$',
            3/2: '$\\frac{3}{2}\\times$',
        }
        fig_time = PlotHelper.plot_times(
            scores, 'PT', xticks_dict=xticks, method_order=method_order,
            db_order=db_order, rename=rename_on_plot, linear=linear)

    fig_folder = get_fig_folder(graphics_folder)

    fig_name = 'boxplots_scores_linear' if linear else 'boxplots_scores'
    fig_time_name = 'boxplots_times_linear' if linear else 'boxplots_times'

    fig.savefig(join(fig_folder, f'{fig_name}.pdf'), bbox_inches='tight')
    fig_time.savefig(join(fig_folder, f'{fig_time_name}.pdf'), bbox_inches='tight')
