"""Prediction tasks v2 for MIMIC."""
import yaml
import os
import pandas as pd
from dask import dataframe as dd

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

# Define overall tables
MIMIC = dbs['MIMIC']
patients = dd.read_csv(MIMIC.frame_paths['patients']).set_index('ROW_ID')
diagnoses_icd = dd.read_csv(MIMIC.frame_paths['diagnoses_icd'], assume_missing=True).set_index('ROW_ID')
patients_diagnosis = patients.merge(diagnoses_icd.drop(['SEQ_NUM'], axis=1), how='left', on='SUBJECT_ID')

# Tasks specific tables
septic_shock = dd.from_pandas(pd.DataFrame({'ICD9_CODE': ['78552']}), npartitions=1)
hemo_shock = dd.from_pandas(pd.DataFrame({'ICD9_CODE': ['78559', '99809', '9584']}), npartitions=1)

# Task 1: Septic shock prediciton
# ----------------------------------------------
# Define y
def define_predict_septic(df):
    """Compute y from patients table."""
    # Ignore given df
    positives = patients_diagnosis.merge(septic_shock, how='inner', on='ICD9_CODE')#.set_index('SUBJECT_ID')
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

# Define which features to keep
septic_pvals_dir = 'pvals/UKBB/septic/'
septic_idx_path = f'{septic_pvals_dir}used_idx.csv'
septic_pvals_path = f'{septic_pvals_dir}pvals_filtered.csv'
if os.path.exists(septic_idx_path) and os.path.exists(septic_pvals_path):
    pvals = pd.read_csv(septic_pvals_path, header=None,
                        index_col=0, squeeze=True)
    pvals = pvals.sort_values()[:n_top_pvals]
    septic_top_pvals = list(pvals.index)

    septic_pvals_keep_transform = Transform(
        output_features=septic_top_pvals
    )

    septic_drop_idx = pd.read_csv(septic_idx_path, index_col=0, squeeze=True)

    septic_idx_transform = Transform(
        input_features=[],
        transform=lambda df: df.drop(septic_drop_idx.index, axis=0),
    )
else:
    septic_pvals_keep_transform = None
    septic_idx_transform = None

task_metas.append(TaskMeta(
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
))
