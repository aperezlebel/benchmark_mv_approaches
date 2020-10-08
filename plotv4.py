"""Plot the results of --train4."""
from prediction.PlotHelperV4 import PlotHelper
import matplotlib.pyplot as plt
import pandas as pd


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
    'UKBB': 'UK BioBank',
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

ph = PlotHelper(root_folder='results_original/graham_v5/results/', rename=rename)
# ph = PlotHelper(root_folder='/Volumes/LACIE/Alexandre/Stage/BACKUP/results_original/graham_v3/results/', rename=rename)

# ph._export('TB', 3)
# exit()

filepath = 'scores/scores.csv'

# # ph.plot_MIA_v_linear(filepath)
# ph.dump(filepath)
# exit()

fig = ph.plot_MIA_linear(filepath, db_order=db_order, method_order=linear_method_order, rename=rename_on_plot)

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
fig = ph.plot_times(filepath, 'PT', xticks_dict=xticks, xlims=(0.005, 15), method_order=linear_method_order, db_order=db_order, rename=rename_on_plot)

# fig = ph.plot_MIA_linear_diff(filepath, db_order=db_order, rename=rename_on_plot)

# plt.tight_layout()
plt.show()
# exit()


# df = ph.get_task_description(filepath)
# df.to_csv('scores/task_description.csv')
# with pd.option_context("max_colwidth", None):
#     df.to_latex('scores/task_description.tex', bold_rows=True)
# print(df)


# exit()

# df = pd.read_csv(filepath)
# df['total_PT'] = df['imputation_PT'].fillna(0) + df['tuning_PT']
# df = PlotHelper.aggregate(df, 'total_PT')
# df = ph._add_relative_value(df, 'total_PT', how='log')
# print(df)
# df.to_csv('sandbox/dump_aggregated_scores.csv')
# exit()
# ph.dump(filepath)
# exit()
# ph.mean_rank(filepath, method_order=method_order).to_csv('scores/ranks.csv')
fig = ph.plot_scores(filepath, method_order=method_order, db_order=db_order, rename=rename_on_plot)#, reference_method='MIA')
if fig:
    plt.show()

xticks = {
    # 0.5: '$\\frac{1}{2}\\times$',
    2/3: '$\\frac{2}{3}\\times$',
    # 0.75: '$\\frac{3}{4}\\times$',
    1: '$1\\times$',
    # 4/3: '$\\frac{4}{3}\\times$',
    3/2: '$\\frac{3}{2}\\times$',
    # 2: '$2\\times$'
}
fig = ph.plot_times(filepath, 'PT', xticks_dict=xticks, method_order=method_order, db_order=db_order, rename=rename_on_plot)
if fig:
    plt.show()
