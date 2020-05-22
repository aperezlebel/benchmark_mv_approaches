"""Prediction tasks v2 for UKBB."""
import os
import yaml
import pandas as pd
import numpy as np

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


# Task 2: platelet prediction (https://arxiv.org/abs/1909.06631)
# --------------------------------------------------------------
platelet_predict_transform = Transform(
    input_features=['Plaquettes'],
    output_features=['Plaquettes'],
)


def define_new_features_platelet(df):
    """Callable used to define new features from a bunch of features."""
    # github.com/wjiang94/ABSLOPE/blob/master/ABSLOPE/OnlineSupp/OnlineSupp.pdf

    print(list(df.columns))
    df = df.astype(float)

    df['Age'] = df['Age du patient (ans)']
    df['SI'] = df['FC en phase hospitalière'].divide(df['Pression Artérielle Systolique - PAS'])
    df['MBP'] = (2*df['Pression Artérielle Diastolique - PAD']+df['Pression Artérielle Systolique - PAS'])/3
    df['Delta.hemo'] = df['Delta Hémocue']
    df['Time.amb'] = df['Délai « arrivée sur les lieux - arrivée hôpital »']
    df['Lactate'] = df['Lactates']
    df['Temp'] = df['Température']
    df['HR'] = df['FC en phase hospitalière']
    df['VE'] = df['Cristalloïdes']+df['Colloïdes']
    df['RBC'] = df['Choc hémorragique (? 4 CGR sur 6h)']
    df['SI.amb'] = df['Fréquence cardiaque (FC) à l arrivée du SMUR'].divide(df['Pression Artérielle Systolique (PAS) à l arrivée du SMUR'])
    df['MAP.amb'] = (2*df['Pression Artérielle Diastolique (PAD) à l arrivée du SMUR']+df['Pression Artérielle Systolique (PAS) à l arrivée du SMUR'])/3
    df['HR.max'] = df['Fréquence cardiaque (FC) maximum']
    df['SBP.min'] = df['Pression Artérielle Systolique (PAS) minimum']
    df['DBP.min'] = df['Pression Artérielle Diastolique (PAD) minimum']

    # Replace potential infinite values by Nans (divide may have created infs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


platelet_new_features_tranform = Transform(
    input_features=[
        'Age du patient (ans)',
        'FC en phase hospitalière',
        'Pression Artérielle Systolique - PAS',
        'Pression Artérielle Diastolique - PAD',
        'Delta Hémocue',
        'Délai « arrivée sur les lieux - arrivée hôpital »',
        'Lactates',
        'Température',
        'FC en phase hospitalière',
        'Cristalloïdes',
        'Colloïdes',
        'Choc hémorragique (? 4 CGR sur 6h)',
        'Fréquence cardiaque (FC) à l arrivée du SMUR',
        'Pression Artérielle Systolique (PAS) à l arrivée du SMUR',
        'Pression Artérielle Diastolique (PAD) à l arrivée du SMUR',
        'Fréquence cardiaque (FC) maximum',
        'Pression Artérielle Systolique (PAS) minimum',
        'Pression Artérielle Diastolique (PAD) minimum',
    ],
    transform=define_new_features_platelet,
    output_features=[
        'Age',
        'SI',
        'MBP',
        'Delta.hemo',
        'Time.amb',
        'Lactate',
        'Temp',
        'HR',
        'VE',
        'RBC',
        'SI.amb',
        'MAP.amb',
        'HR.max',
        'SBP.min',
        'DBP.min',
    ],
)

task_metas.append(TaskMeta(
    name='platelet',
    db='TB',
    df_name='20000',
    classif=False,
    idx_selection=None,
    predict=platelet_predict_transform,
    transform=platelet_new_features_tranform,
    select=None,
    encode='ordinal',
))
