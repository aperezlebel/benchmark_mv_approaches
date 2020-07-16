"""Plot the results of --train4."""
from prediction.PlotHelperV4 import PlotHelperV4
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
}

rename_on_plot = {
    'relative_score': 'Relative prediction score',
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

db_order = [
    'TB',
    'UKBB',
    'MIMIC',
    'NHIS',
]

ph = PlotHelperV4(root_folder='results_original/graham_v4/results/', rename=rename)
# ph = PlotHelperV4(root_folder='/Volumes/LACIE/Alexandre/Stage/BACKUP/results_original/graham_v3/results/', rename=rename)

# ph._export('TB', 3)
# exit()

filepath = 'scores/scores.csv'
df = ph.get_task_description(filepath)
df.to_csv('scores/task_description.csv')
with pd.option_context("max_colwidth", None):
    df.to_latex('scores/task_description.tex')
print(df)
exit()

# df = pd.read_csv(filepath)
# df['total_PT'] = df['imputation_PT'].fillna(0) + df['tuning_PT']
# df = PlotHelperV4.aggregate(df, 'total_PT')
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
