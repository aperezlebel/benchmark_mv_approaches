"""Prediction tasks v2 for NHIS."""
import os
import pandas as pd

from .task_v2 import TaskMeta
from .transform import Transform


# Task 1: income prediciton
# -------------------------------
def income_task(**kwargs):
    """Return TaskMeta for income prediction."""
    income_predict_transform = Transform(
        input_features=['ERNYR-P'],
        output_features=['ERNYR-P'],
    )

    # Drop features linked to feature to predict
    income_drop_features = {
        'INCGRP4',
        'INCGRP5',
    }

    assert 'n_top_pvals' in kwargs
    n_top_pvals = kwargs['n_top_pvals']

    if n_top_pvals is None:
        income_pvals_keep_transform = None
        income_idx_transform = None

    else:

        assert 'RS' in kwargs
        assert 'T' in kwargs

        RS = kwargs['RS']
        T = kwargs['T']
        income_pvals_dir = 'pvals/NHIS/income_pvals/'
        income_idx_path = f'{income_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        income_pvals_path = f'{income_pvals_dir}RS{RS}-T{T}-pvals.csv'

        assert os.path.exists(income_idx_path)
        assert os.path.exists(income_pvals_path)

        pvals = pd.read_csv(income_pvals_path, header=None,
                            index_col=0, squeeze=True)

        # Match exact feature or start with and followed by '_' (categorical)
        for f in income_drop_features:  # Drop asked features from pvals
            regex = f'(^{f}$|^{f}_)'
            pvals = pvals[~pvals.index.str.match(regex)]
        pvals = pvals.sort_values()[:n_top_pvals]
        income_top_pvals = list(pvals.index.astype(str))

        income_pvals_keep_transform = Transform(
            output_features=income_top_pvals
        )

        income_drop_idx = pd.read_csv(income_idx_path, index_col=0,
                                      squeeze=True)

        income_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(income_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='income_pvals',
        db='NHIS',
        df_name='X_income',
        classif=True,
        idx_column='IDX',
        idx_selection=income_idx_transform,
        predict=income_predict_transform,
        transform=None,
        select=income_pvals_keep_transform,
        encode_select='all',
        encode_transform=None,
    )


# Task 2: BMI prediciton
# ----------------------
def bmi_task(**kwargs):
    """Return TaskMeta for bmi prediction."""
    bmi_predict_transform = Transform(
        input_features=['BMI'],
        output_features=['BMI'],
    )

    # Drop features linked to feature to predict
    bmi_drop_features = {
        'BMI',
        'AHEIGHT',
        'AWEIGHTP',
    }

    assert 'n_top_pvals' in kwargs
    n_top_pvals = kwargs['n_top_pvals']

    if n_top_pvals is None:
        bmi_pvals_keep_transform = None
        bmi_idx_transform = None

    else:

        assert 'RS' in kwargs
        assert 'T' in kwargs

        RS = kwargs['RS']
        T = kwargs['T']
        bmi_pvals_dir = 'pvals/NHIS/bmi_pvals/'
        bmi_idx_path = f'{bmi_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        bmi_pvals_path = f'{bmi_pvals_dir}RS{RS}-T{T}-pvals.csv'

        assert os.path.exists(bmi_idx_path)
        assert os.path.exists(bmi_pvals_path)

        pvals = pd.read_csv(bmi_pvals_path, header=None,
                            index_col=0, squeeze=True)

        # Match exact feature or start with and followed by '_' (categorical)
        for f in bmi_drop_features:  # Drop asked features from pvals
            regex = f'(^{f}$|^{f}_)'
            pvals = pvals[~pvals.index.str.match(regex)]
        pvals = pvals.sort_values()[:n_top_pvals]
        bmi_top_pvals = list(pvals.index.astype(str))
        print(bmi_top_pvals)

        bmi_pvals_keep_transform = Transform(
            output_features=bmi_top_pvals
        )

        bmi_drop_idx = pd.read_csv(bmi_idx_path, index_col=0, squeeze=True)

        bmi_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(bmi_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='bmi_pvals',
        db='NHIS',
        df_name='X_income',
        classif=False,
        idx_column='IDX',
        idx_selection=bmi_idx_transform,
        predict=bmi_predict_transform,
        transform=None,
        select=bmi_pvals_keep_transform,
        encode_select='all',
        encode_transform=None,
    )


task_metas = {
    'income_pvals': income_task,
    'bmi_pvals': bmi_task,
}
