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
    encode_select='all',
    encode_transform=None,
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
        # 'FC en phase hospitalière',
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
    encode_select='all',
    encode_transform='ordinal',
))


# Task 3: Hemorrhagic shock prediciton (https://arxiv.org/pdf/1805.04602)
# -----------------------------------------------------------------------
shock_hemo_predict_transform = Transform(
    input_features=['Choc hémorragique (? 4 CGR sur 6h)'],
    output_features=['Choc hémorragique (? 4 CGR sur 6h)'],
)


def define_new_features_shock_hemo(df):
    """Callable used to define new features from a bunch of features."""
    df = df.astype(float)

    df['Age'] = df['Age du patient (ans)']
    df['BMI'] = df['BMI']
    df['FC.SMUR'] = df['Fréquence cardiaque (FC) à l arrivée du SMUR']
    df['SD.SMUR'] = df['Pression Artérielle Systolique (PAS) à l arrivée du SMUR'] - df['Pression Artérielle Diastolique (PAD) à l arrivée du SMUR']
    df['SD.min'] = df['Pression Artérielle Systolique (PAS) minimum'] - df['Pression Artérielle Diastolique (PAD) minimum']
    df['FC.max'] = df['Fréquence cardiaque (FC) maximum']
    df['Glasgow.moteur.init'] = df['Glasgow moteur initial']
    df['Glasgow.init'] = df['Glasgow initial']
    df['Hemocue.init'] = df['Hémocue initial']
    df['SpO2.min'] = df['SpO2 min']
    df['RT.colloides'] = df['Colloïdes']
    df['RT.cristalloides'] = df['Cristalloïdes']

    # Replace potential infinite values by Nans (divide may have created infs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


shock_hemo_new_features_tranform = Transform(
    input_features=[
        'Age du patient (ans)',
        'BMI',
        'Fréquence cardiaque (FC) à l arrivée du SMUR',
        'Pression Artérielle Systolique (PAS) à l arrivée du SMUR',
        'Pression Artérielle Diastolique (PAD) à l arrivée du SMUR',
        'Pression Artérielle Systolique (PAS) minimum',
        'Pression Artérielle Diastolique (PAD) minimum',
        'Fréquence cardiaque (FC) maximum',
        'Glasgow moteur initial',
        'Glasgow initial',
        'Hémocue initial',
        'SpO2 min',
        'Colloïdes',
        'Cristalloïdes',
    ],
    transform=define_new_features_shock_hemo,
    output_features=[
        'Age',
        'BMI',
        'FC.SMUR',
        'SD.SMUR',
        'SD.min',
        'FC.max',
        'Glasgow.moteur.init',
        'Glasgow.init',
        'Hemocue.init',
        'SpO2.min',
        'RT.colloides',
        'RT.cristalloides'
    ],
)

task_metas.append(TaskMeta(
    name='shock_hemo',
    db='TB',
    df_name='20000',
    classif=True,
    idx_selection=None,
    predict=shock_hemo_predict_transform,
    transform=shock_hemo_new_features_tranform,
    select=None,
    encode_transform=None,
    encode_select=None,
))

# Task 4: Tranexamic acid prediction (https://arxiv.org/abs/1910.10624)
# ---------------------------------------------------------------------
acid_predict_transform = Transform(
    input_features=['Acide tranexamique'],
    output_features=['Acide tranexamique'],
)


def define_new_features_acid(df):
    """Callable used to define new features from a bunch of features."""
    df = df.astype(float)

    # Temp features (will be dropped)
    df['SBP.min'] = df['Pression Artérielle Systolique (PAS) minimum']
    df['SBP.MICU'] = df['Pression Artérielle Systolique (PAS) à l arrivée du SMUR']
    df['DBP.min'] = df['Pression Artérielle Diastolique (PAD) minimum']
    df['DBP.MICU'] = df['Pression Artérielle Diastolique (PAD) à l arrivée du SMUR']
    df['HR.max'] = df['Fréquence cardiaque (FC) maximum']
    df['HR.MICU'] = df['Fréquence cardiaque (FC) à l arrivée du SMUR']
    df['Shock.index.h'] = df['FC en phase hospitalière'].divide(df['Pression Artérielle Systolique - PAS'])

    # Persistent features
    df['SBP.ph'] = np.minimum(df['SBP.min'], df['SBP.MICU'])
    df['DBP.ph'] = np.minimum(df['DBP.min'], df['DBP.MICU'])
    df['HR.ph'] = np.maximum(df['HR.max'], df['HR.MICU'])
    df['Cardiac.arrest.ph'] = df['Arrêt cardio-respiratoire (massage)']
    df['HemoCue.init'] = df['Hémocue initial']
    df['SpO2.min'] = df['SpO2 min']
    df['Vasopressor.therapy'] = df['Catécholamines max dans choc hémorragique']
    df['Cristalloid.volume'] = df['Cristalloïdes']
    df['Colloid.volume'] = df['Colloïdes']
    df['Shock.index.ph'] = df['Fréquence cardiaque (FC) à l arrivée du SMUR'].divide(df['Pression Artérielle Systolique (PAS) à l arrivée du SMUR'])
    df['AIS.external'] = df['ISS  / External']
    df['Delta.shock.index'] = df['Shock.index.h'] - df['Shock.index.ph']
    df['Delta.hemoCue'] = df['Delta Hémocue']

    df['Anticoagulant.therapy'] = df['Traitement anticoagulant']
    df['Antiplatelet.therapy'] = df['Traitement antiagrégants']
    df['GCS.init'] = df['Glasgow initial']
    df['GCS'] = df['Score de Glasgow en phase hospitalière']
    df['GCS.motor.init'] = df['Glasgow moteur initial']
    df['GCS.motor'] = df['Glasgow moteur']
    df['Improv.anomaly.osmo'] = df['Régression mydriase sous osmothérapie']
    df['Medcare.time.ph'] = df['Délai « arrivée sur les lieux - arrivée hôpital »']
    df['FiO2'] = df['FiO2']
    df['Temperature.min'] = df['Température min']
    df['TCD.PI.max'] = df['DTC IP max (sur les premières 24 heures d HTIC)']
    df['IICP'] = df['HTIC (>25 PIC simple sédation)']
    df['EVD'] = df['Dérivation ventriculaire externe (DVE)']
    df['Decompressive.craniectomy'] = df['Craniectomie dé-compressive']
    df['Neurosurgery.day0'] = df['Bloc dans les premières 24h  / Neurochirurgie (ex. : Craniotomie ou DVE)']
    df['AIS.head'] = df['ISS  / Head_neck']
    df['AIS.face'] = df['ISS  / Face']
    df['ISS'] = df['Score ISS']
    df['ISS.II'] = df['Total Score IGS']

    # Replace potential infinite values by Nans (divide may have created infs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


acid_new_features_tranform = Transform(
    input_features=[
        'Pression Artérielle Systolique (PAS) minimum',
        'Pression Artérielle Systolique (PAS) à l arrivée du SMUR',
        'Pression Artérielle Diastolique (PAD) minimum',
        'Pression Artérielle Diastolique (PAD) à l arrivée du SMUR',
        'Fréquence cardiaque (FC) maximum',
        'Fréquence cardiaque (FC) à l arrivée du SMUR',
        'FC en phase hospitalière',
        'Pression Artérielle Systolique - PAS',
        # 'Numéro de centre',
        'Arrêt cardio-respiratoire (massage)',
        'Hémocue initial',
        'SpO2 min',
        'Catécholamines max dans choc hémorragique',
        'Cristalloïdes',
        'Colloïdes',
        'ISS  / External',
        'Delta Hémocue',
        'Traitement anticoagulant',
        'Traitement antiagrégants',
        'Glasgow initial',
        'Score de Glasgow en phase hospitalière',
        'Glasgow moteur initial',
        'Glasgow moteur',
        # 'Anomalie pupillaire (Pré-hospitalier)',
        # 'Anomalie pupillaire (Phase hospitalière)',
        # 'Osmothérapie',
        'Régression mydriase sous osmothérapie',
        'Délai « arrivée sur les lieux - arrivée hôpital »',
        'FiO2',
        'Température min',
        'DTC IP max (sur les premières 24 heures d HTIC)',
        'HTIC (>25 PIC simple sédation)',
        'Dérivation ventriculaire externe (DVE)',
        'Craniectomie dé-compressive',
        'Bloc dans les premières 24h  / Neurochirurgie (ex. : Craniotomie ou DVE)',
        'ISS  / Head_neck',
        'ISS  / Face',
        'Score ISS',
        'Total Score IGS',
    ],
    transform=define_new_features_acid,
    output_features=[
        # 'Trauma.center',
        'SBP.ph',
        'DBP.ph',
        'HR.ph',
        'Cardiac.arrest.ph',
        'HemoCue.init',
        'SpO2.min',
        'Vasopressor.therapy',
        'Cristalloid.volume',
        'Colloid.volume',
        'Shock.index.ph',
        'AIS.external',
        'Delta.shock.index',
        'Delta.hemoCue',
        'Anticoagulant.therapy',
        'Antiplatelet.therapy',
        'GCS.init',
        'GCS',
        'GCS.motor.init',
        'GCS.motor',
        # 'Pupil.anomaly.ph',
        # 'Pupil.anomaly.h',
        # 'Osmotherapy',
        'Improv.anomaly.osmo',
        'Medcare.time.ph',
        'FiO2',
        'Temperature.min',
        'TCD.PI.max',
        'IICP',
        'EVD',
        'Decompressive.craniectomy',
        'Neurosurgery.day0',
        'AIS.head',
        'AIS.face',
        'ISS',
        'ISS.II',
    ],
)

acid_keep_transform = Transform(
    input_features=[
        'Numéro de centre',
        'Anomalie pupillaire (Pré-hospitalier)',
        'Anomalie pupillaire (Phase hospitalière)',
        'Osmothérapie',
    ],
)

task_metas.append(TaskMeta(
    name='acid',
    db='TB',
    df_name='20000',
    classif=True,
    idx_selection=None,
    predict=acid_predict_transform,
    transform=acid_new_features_tranform,
    select=acid_keep_transform,
    encode_transform='ordinal',
    encode_select='all',
))
