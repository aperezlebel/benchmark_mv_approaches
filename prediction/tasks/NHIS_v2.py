"""Prediction tasks v2 for NHIS."""
import yaml
import os
import pandas as pd

from .task_v2 import TaskMeta
from .transform import Transform
from database import dbs


task_metas = list()

# Load some params from custom file
filepath = 'custom/strategy_params.yml'
if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
else:
    params = dict()

n_top_pvals = params.get('n_top_pvals', 100)


# Task 1: income shock prediciton
# ----------------------------------------------
income_predict_transform = Transform(
    input_features=['ERNYR_P'],
    output_features=['ERNYR_P'],
)

# Drop features linked to feature to predict
income_drop_features = {
    'INCGRP4',
    'INCGRP5',
}

# Define which features to keep
income_pvals_dir = 'pvals/NHIS/income/'
income_idx_path = f'{income_pvals_dir}used_idx.csv'
income_pvals_path = f'{income_pvals_dir}pvals_filtered.csv'
if os.path.exists(income_idx_path) and os.path.exists(income_pvals_path):
    pvals = pd.read_csv(income_pvals_path, header=None,
                        index_col=0, squeeze=True)
    pvals = pvals.sort_values()[:n_top_pvals]
    income_top_pvals = list(pvals.index)

    income_pvals_keep_transform = Transform(
        output_features=income_top_pvals
    )

    income_drop_idx = pd.read_csv(income_idx_path, index_col=0, squeeze=True)

    income_idx_transform = Transform(
        input_features=[],
        transform=lambda df: df.drop(income_drop_idx.index, axis=0),
    )
else:
    income_pvals_keep_transform = None
    income_idx_transform = None

task_metas.append(TaskMeta(
    name='income_pvals',
    db='NHIS',
    df_name='X_income',
    classif=True,
    idx_column=None,
    idx_selection=income_idx_transform,
    predict=income_predict_transform,
    transform=None,
    select=income_pvals_keep_transform,
    encode_select=None,
    encode_transform=None,
    drop=income_drop_features,
))
