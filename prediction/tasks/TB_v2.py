"""Prediction tasks v2 for UKBB."""
import os
import yaml
import pandas as pd

from .task_v2 import TaskMeta
from .transform import Transform


task_metas = list()

# Load some params from custom file
filepath = 'custom/strategy_params.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
else:
    params = dict()

n_top_pvals = params.get('n_top_pvals', 100)


# Task 1: Death prediction
# ------------------------
death_predict_transform = Transform(
    input_features=['Décès'],
    transform=None,
    output_features=['Décès'],
)

death_pvals_path = 'pvals/TB/death/pvals_filtered.csv'
if os.path.exists(death_pvals_path):
    pvals = pd.read_csv(death_pvals_path, header=None,
                        index_col=0, squeeze=True)
    pvals = pvals.sort_values()[:n_top_pvals]
    breast_top_pvals = list(pvals.index)

    breast_pvals_keep_transform = Transform(
        output_features=breast_top_pvals
    )
else:
    breast_pvals_keep_transform = None

task_metas.append(TaskMeta(
    name='death_pvals',
    db='TB',
    df_name='20000',
    classif=True,
    idx_selection=None,
    predict=death_predict_transform,
    transform=None,
    select=None,
    encode='all',
))
