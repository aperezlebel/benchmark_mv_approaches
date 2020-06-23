"""Prediction tasks v2 for MIMIC."""
import os
import pandas as pd
from dask import dataframe as dd

from .task_v2 import TaskMeta
from .transform import Transform
from database import dbs


# Define overall tables
MIMIC = dbs['MIMIC']
patients = dd.read_csv(MIMIC.frame_paths['patients']).set_index('ROW_ID')
diagnoses_icd = dd.read_csv(MIMIC.frame_paths['diagnoses_icd'], assume_missing=True).set_index('ROW_ID')
patients_diagnosis = patients.merge(diagnoses_icd.drop(['SEQ_NUM'], axis=1), how='left', on='SUBJECT_ID')

# Tasks specific tables
septic_shock = dd.from_pandas(pd.DataFrame({'ICD9_CODE': ['78552']}), npartitions=1)
hemo_shock = dd.from_pandas(pd.DataFrame({'ICD9_CODE': ['78559', '99809', '9584']}), npartitions=1)


# Task 1: Septic shock prediciton
# -------------------------------
def septic_task(**kwargs):
    """Return TaskMeta for septic shock prediction."""
    # Define y
    def define_predict_septic(df):
        """Compute y from patients table."""
        # Ignore given df
        positives = patients_diagnosis.merge(septic_shock, how='inner', on='ICD9_CODE')
        positives = positives.drop_duplicates(subset=['SUBJECT_ID']).set_index('SUBJECT_ID').index
        positives_idx = positives.compute()

        # Get full idx from df and set the complementary to 0
        # idx = patients.set_index('SUBJECT_ID').index.compute()
        idx = df.index
        # need to intersect because one index of positives_idx is not in idx
        negatives_idx = idx.difference(positives_idx).intersection(idx)
        positives_idx = positives_idx.intersection(idx)

        positives = pd.DataFrame({'y': 1}, index=positives_idx)
        negatives = pd.DataFrame({'y': 0}, index=negatives_idx)
        df = pd.concat((positives, negatives), axis=0).sort_index()

        return df

    septic_predict_transform = Transform(
        input_features=[],
        transform=define_predict_septic,
        output_features=['y'],
    )

    assert 'n_top_pvals' in kwargs
    n_top_pvals = kwargs['n_top_pvals']

    if n_top_pvals is None:
        septic_pvals_keep_transform = None
        septic_idx_transform = None

    else:

        assert 'RS' in kwargs
        assert 'T' in kwargs

        RS = kwargs['RS']
        T = kwargs['T']
        septic_pvals_dir = 'pvals/MIMIC/septic_pvals/'
        septic_idx_path = f'{septic_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        septic_pvals_path = f'{septic_pvals_dir}RS{RS}-T{T}-pvals.csv'

        assert os.path.exists(septic_idx_path)
        assert os.path.exists(septic_pvals_path)

        pvals = pd.read_csv(septic_pvals_path, header=None,
                            index_col=0, squeeze=True)

        pvals = pvals.sort_values()[:n_top_pvals]
        septic_top_pvals = list(pvals.index.astype(str))

        septic_pvals_keep_transform = Transform(
            output_features=septic_top_pvals
        )

        septic_drop_idx = pd.read_csv(septic_idx_path, index_col=0,
                                      squeeze=True)

        septic_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(septic_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='septic_pvals',
        db='MIMIC',
        df_name='X_labevents',
        classif=True,
        idx_column='subject_id',
        idx_selection=septic_idx_transform,
        predict=septic_predict_transform,
        transform=None,
        select=septic_pvals_keep_transform,
        encode_select=None,
        encode_transform=None,
    )


# Task 2: Hemorrhagic shock prediciton
# ------------------------------------
def hemo_task(**kwargs):
    """Return TaskMeta for Hemorrhagic shock prediction."""
    # Define y
    def define_predict_hemo(df):
        """Compute y from patients table."""
        # Ignore given df
        positives = patients_diagnosis.merge(hemo_shock, how='inner', on='ICD9_CODE')
        positives = positives.drop_duplicates(subset=['SUBJECT_ID']).set_index('SUBJECT_ID').index
        positives_idx = positives.compute()

        # Get full idx from df and set the complementary to 0
        # idx = patients.set_index('SUBJECT_ID').index.compute()
        idx = df.index
        # need to intersect because one index of positives_idx is not in idx
        negatives_idx = idx.difference(positives_idx).intersection(idx)
        positives_idx = positives_idx.intersection(idx)

        positives = pd.DataFrame({'y': 1}, index=positives_idx)
        negatives = pd.DataFrame({'y': 0}, index=negatives_idx)
        df = pd.concat((positives, negatives), axis=0).sort_index()

        return df

    hemo_predict_transform = Transform(
        input_features=[],
        transform=define_predict_hemo,
        output_features=['y'],
    )

    assert 'n_top_pvals' in kwargs
    n_top_pvals = kwargs['n_top_pvals']

    if n_top_pvals is None:
        hemo_pvals_keep_transform = None
        hemo_idx_transform = None

    else:

        assert 'RS' in kwargs
        assert 'T' in kwargs

        RS = kwargs['RS']
        T = kwargs['T']
        hemo_pvals_dir = 'pvals/MIMIC/hemo_pvals/'
        hemo_idx_path = f'{hemo_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        hemo_pvals_path = f'{hemo_pvals_dir}RS{RS}-T{T}-pvals.csv'

        assert os.path.exists(hemo_idx_path)
        assert os.path.exists(hemo_pvals_path)

        pvals = pd.read_csv(hemo_pvals_path, header=None,
                            index_col=0, squeeze=True)

        pvals = pvals.sort_values()[:n_top_pvals]
        hemo_top_pvals = list(pvals.index.astype(str))

        hemo_pvals_keep_transform = Transform(
            output_features=hemo_top_pvals
        )

        hemo_drop_idx = pd.read_csv(hemo_idx_path, index_col=0, squeeze=True)

        hemo_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(hemo_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='hemo_pvals',
        db='MIMIC',
        df_name='X_labevents',
        classif=True,
        idx_column='subject_id',
        idx_selection=hemo_idx_transform,
        predict=hemo_predict_transform,
        transform=None,
        select=hemo_pvals_keep_transform,
        encode_select=None,
        encode_transform=None,
    )


task_metas = {
    'septic_pvals': septic_task,
    'hemo_pvals': hemo_task,
}
