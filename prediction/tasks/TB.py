"""Prediction tasks for UKBB."""
import os
import pandas as pd
import numpy as np

from .task import TaskMeta
from .transform import Transform


# Task 1: Death prediction
# ------------------------
def death_task(**kwargs):
    """Return TaskMeta for death prediction."""
    death_predict_transform = Transform(
        input_features=['Décès'],
        output_features=['Décès'],
    )

    assert 'n_top_pvals' in kwargs
    n_top_pvals = kwargs['n_top_pvals']

    if n_top_pvals is None:
        death_pvals_keep_transform = None
        death_idx_transform = None

    else:

        assert 'RS' in kwargs
        assert 'T' in kwargs

        RS = kwargs['RS']
        T = kwargs['T']
        death_pvals_dir = 'pvals/TB/death_pvals/'
        death_idx_path = f'{death_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        death_pvals_path = f'{death_pvals_dir}RS{RS}-T{T}-pvals_filtered.csv'

        assert os.path.exists(death_idx_path)
        assert os.path.exists(death_pvals_path)

        pvals = pd.read_csv(death_pvals_path, header=None,
                            index_col=0, squeeze=True)

        pvals = pvals.sort_values()[:n_top_pvals]
        death_top_pvals = list(pvals.index.astype(str))

        death_pvals_keep_transform = Transform(
            output_features=death_top_pvals
        )

        death_drop_idx = pd.read_csv(death_idx_path, index_col=0, squeeze=True)

        death_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(death_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='death_pvals',
        db='TB',
        df_name='20000',
        classif=True,
        idx_column='ID_PATIENT',
        idx_selection=death_idx_transform,
        predict=death_predict_transform,
        transform=None,
        select=death_pvals_keep_transform,
        encode_select='all',
        encode_transform=None,
    )


# Task 2: platelet prediction (https://arxiv.org/abs/1909.06631)
# --------------------------------------------------------------
# This is outside the callable because used in the pvals version of the task
platelet_predict_transform = Transform(
    input_features=['Plaquettes'],
    output_features=['Plaquettes'],
)


def platelet_task(**kwargs):
    """Return TaskMeta for platelet prediction."""
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

    return TaskMeta(
        name='platelet',
        db='TB',
        df_name='20000',
        classif=False,
        idx_column='ID_PATIENT',
        idx_selection=None,
        predict=platelet_predict_transform,
        transform=platelet_new_features_tranform,
        select=None,
        encode_select='all',
        encode_transform='ordinal',
    )


# Task 2.2: platelet prediction using pvals
# -----------------------------------------
def platelet_pvals_task(**kwargs):
    """Return TaskMeta for platelet prediction."""
    # Drop features linked to feature to predict
    platelet_drop_features = {
        'Plaquettes',
    }

    assert 'n_top_pvals' in kwargs
    n_top_pvals = kwargs['n_top_pvals']

    if n_top_pvals is None:
        platelet_pvals_keep_transform = None
        platelet_idx_transform = None

    else:

        assert 'RS' in kwargs
        assert 'T' in kwargs

        RS = kwargs['RS']
        T = kwargs['T']
        platelet_pvals_dir = 'pvals/TB/platelet_pvals/'
        platelet_idx_path = f'{platelet_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        platelet_pvals_path = f'{platelet_pvals_dir}RS{RS}-T{T}-pvals_filtered.csv'

        assert os.path.exists(platelet_idx_path)
        assert os.path.exists(platelet_pvals_path)

        pvals = pd.read_csv(platelet_pvals_path, header=None,
                            index_col=0, squeeze=True)

        # Match exact feature or start with and followed by '_' (categorical)
        for f in platelet_drop_features:  # Drop asked features from pvals
            regex = f'(^{f}$|^{f}_)'
            pvals = pvals[~pvals.index.str.match(regex)]
        pvals = pvals.sort_values()[:n_top_pvals]
        platelet_top_pvals = list(pvals.index.astype(str))

        platelet_pvals_keep_transform = Transform(
            output_features=platelet_top_pvals
        )

        platelet_drop_idx = pd.read_csv(platelet_idx_path, index_col=0, squeeze=True)

        platelet_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(platelet_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='platelet_pvals',
        db='TB',
        df_name='20000',
        classif=False,
        idx_column='ID_PATIENT',
        idx_selection=platelet_idx_transform,
        predict=platelet_predict_transform,
        transform=None,
        select=platelet_pvals_keep_transform,
        encode_select='all',
        encode_transform=None,
    )


# Task 3: Hemorrhagic shock prediciton (https://arxiv.org/pdf/1805.04602)
# -----------------------------------------------------------------------
# This is outside the callable because used in the pvals version of the task
hemo_predict_transform = Transform(
    input_features=['Choc hémorragique (? 4 CGR sur 6h)'],
    output_features=['Choc hémorragique (? 4 CGR sur 6h)'],
)


def hemo_task(**kwargs):
    """Return TaskMeta for hemo shock prediction."""

    def define_new_features_hemo(df):
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

    hemo_new_features_tranform = Transform(
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
        transform=define_new_features_hemo,
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

    return TaskMeta(
        name='hemo',
        db='TB',
        df_name='20000',
        classif=True,
        idx_column='ID_PATIENT',
        idx_selection=None,
        predict=hemo_predict_transform,
        transform=hemo_new_features_tranform,
        select=None,
        encode_transform=None,
        encode_select=None,
    )


# Task 3.2: Hemorrhagic shock prediciton using pvals
# --------------------------------------------------
def hemo_pvals_task(**kwargs):
    """Return TaskMeta for hemo shock prediction."""
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
        hemo_pvals_dir = 'pvals/TB/hemo_pvals/'
        hemo_idx_path = f'{hemo_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        hemo_pvals_path = f'{hemo_pvals_dir}RS{RS}-T{T}-pvals_filtered.csv'

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
        db='TB',
        df_name='20000',
        classif=True,
        idx_column='ID_PATIENT',
        idx_selection=hemo_idx_transform,
        predict=hemo_predict_transform,
        transform=None,
        select=hemo_pvals_keep_transform,
        encode_transform=None,
        encode_select='all',
    )


# Task 4: Tranexamic acid prediction (https://arxiv.org/abs/1910.10624)
# ---------------------------------------------------------------------
def acid_task(**kwargs):
    """Return TaskMeta for acid prediction."""
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
        df['AIS.head'] = df['ISS  / Head neck']
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
            'ISS  / Head neck',
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

    return TaskMeta(
        name='acid',
        db='TB',
        df_name='20000',
        classif=True,
        idx_column='ID_PATIENT',
        idx_selection=None,
        predict=acid_predict_transform,
        transform=acid_new_features_tranform,
        select=acid_keep_transform,
        encode_transform='ordinal',
        encode_select='all',
    )


# Task 5: Septic shock prediction
# -------------------------------
def septic_task(**kwargs):
    """Return TaskMeta for septic prediction."""
    septic_predict_transform = Transform(
        input_features=['Choc septique'],
        output_features=['Choc septique'],
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
        septic_pvals_dir = 'pvals/TB/septic_pvals/'
        septic_idx_path = f'{septic_pvals_dir}RS{RS}-T{T}-used_idx.csv'
        septic_pvals_path = f'{septic_pvals_dir}RS{RS}-T{T}-pvals_filtered.csv'

        assert os.path.exists(septic_idx_path)
        assert os.path.exists(septic_pvals_path)

        pvals = pd.read_csv(septic_pvals_path, header=None,
                            index_col=0, squeeze=True)

        assert 'n_top_pvals' in kwargs
        n_top_pvals = kwargs['n_top_pvals']
        pvals = pvals.sort_values()[:n_top_pvals]
        septic_top_pvals = list(pvals.index.astype(str))

        septic_pvals_keep_transform = Transform(
            output_features=septic_top_pvals
        )

        septic_drop_idx = pd.read_csv(septic_idx_path,
                                      index_col=0, squeeze=True)

        septic_idx_transform = Transform(
            input_features=[],
            transform=lambda df: df.drop(septic_drop_idx.index, axis=0),
        )

    return TaskMeta(
        name='septic_pvals',
        db='TB',
        df_name='20000',
        classif=True,
        idx_column='ID_PATIENT',
        idx_selection=septic_idx_transform,
        predict=septic_predict_transform,
        transform=None,
        select=septic_pvals_keep_transform,
        encode_select='all',
        encode_transform=None,
    )


task_metas = {
    'death_pvals': death_task,
    # 'platelet': platelet_task,
    'platelet_pvals': platelet_pvals_task,
    'hemo': hemo_task,
    'hemo_pvals': hemo_pvals_task,
    # 'acid': acid_task,
    'septic_pvals': septic_task,
}
