"""Build prediction tasks for the TB database."""
import numpy as np

from df_utils import get_columns

from .taskMeta import TaskMeta

tasks_meta = list()


# Task 1: Death prediction
def transform_df_death(df, **kwargs):
    predict = kwargs['meta'].predict
    # Drop rows with missing values in the feature to predict
    return df.dropna(axis=0, subset=[predict])


tasks_meta.append(TaskMeta(
    name='death',
    db='TB',
    df_name='20000',
    predict="Décès",
    drop=[
        "Date de décès (à l'hôpital après sortie de réanimation)",
        "Cause du décès_A survécu",
        "Cause du décès_Autre (précisez ci-dessous)",
        "Cause du décès_Choc hémorragique",
        "Cause du décès_Choc septique",
        "Cause du décès_Défaillance multi-viscérale",
        "Cause du décès_LATA",
        "Cause du décès_Mort encéphalique",
        "Cause du décès_Trauma cranien",
        "Cause du décès_z MISSING_VALUE",
        "Transfert secondaire, pourquoi ?_Pas de transfert",
        "Transfert secondaire, pourquoi ?_Plateau technique insuffisant",
        "Transfert secondaire, pourquoi ?_Rapprochement familial",
        "Transfert secondaire, pourquoi ?_z MISSING_VALUE",
        "Sortie_Autre réanimation",
        "Sortie_Centre de rééducation",
        "Sortie_Domicile",
        "Sortie_Service hospitalier",
        "Sortie_z MISSING_VALUE",
        "Glasgow de sortie",
        "Nombre de jours à l'hôpital",
        "Durée de séjour en réa- si date de sortie connue, durée de séjour = (date sortie - date d entrée)- si date de sortie inconnue, d",
        "Nombre de jours de VM",
        "Procédure limitations de soins (LATA)",
    ],
    transform=transform_df_death
))


# Task 2: Platelet prediciton (https://arxiv.org/abs/1909.06631)
def transform_df_platelet(df, **kwargs):
    """Build df with appropiate features for platelet prediciton following.

    github.com/wjiang94/ABSLOPE/blob/master/ABSLOPE/OnlineSupp/OnlineSupp.pdf
    """
    predict = kwargs['meta'].predict

    df = df.copy()
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

    # Drop rows with missing values in the feature to predict
    return df.dropna(axis=0, subset=[predict])


tasks_meta.append(TaskMeta(
    name='platelet',
    db='TB',
    df_name='20000',
    predict='Plaquettes',
    keep=[
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
        'DBP.min'
    ],
    transform=transform_df_platelet
))


# Task 3: Hemorrhagic shock prediciton (https://arxiv.org/pdf/1805.04602)
def transform_df_shock_hemo(df, **kwargs):
    """Build df with appropiate features for Hemmoohagic shock prediction."""
    predict = kwargs['meta'].predict

    df = df.copy()
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

    # Drop rows with missing values in the feature to predict
    # df[df_mv[predict] != 0] = np.nan
    return df.dropna(axis=0, subset=[predict])


tasks_meta.append(TaskMeta(
    name='shock_hemo',
    db='TB',
    df_name='20000',
    predict='Choc hémorragique (? 4 CGR sur 6h)',
    keep=[
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
    transform=transform_df_shock_hemo
))

# Task 4: Tranexamic acid prediction (https://arxiv.org/abs/1910.10624)
rename_acid = {
    'Numéro de centre': 'Trauma.center',
    'Anomalie pupillaire (Pré-hospitalier)': 'Pupil.anomaly.ph',
    'Anomalie pupillaire (Phase hospitalière)': 'Pupil.anomaly.h',
    'Osmothérapie': 'Osmotherapy',
}


def transform_df_acid(df, **kwargs):
    """Build df with appropiate features for tranexamic acid prediction."""
    predict = kwargs['meta'].predict

    df = df.copy()

    # Temp features (will be dropped)
    df['SBP.min'] = df['Pression Artérielle Systolique (PAS) minimum']
    df['SBP.MICU'] = df['Pression Artérielle Systolique (PAS) à l arrivée du SMUR']
    df['DBP.min'] = df['Pression Artérielle Diastolique (PAD) minimum']
    df['DBP.MICU'] = df['Pression Artérielle Diastolique (PAD) à l arrivée du SMUR']
    df['HR.max'] = df['Fréquence cardiaque (FC) maximum']
    df['HR.MICU'] = df['Fréquence cardiaque (FC) à l arrivée du SMUR']
    df['Shock.index.h'] = df['FC en phase hospitalière'].divide(df['Pression Artérielle Systolique - PAS'])

    # Persistent features
    # df['Trauma.center'] = df['Numéro de centre']
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
    # df['Pupil.anomaly.ph'] = df['Anomalie pupillaire (Pré-hospitalier)']
    # df['Pupil.anomaly.h'] = df['Anomalie pupillaire (Phase hospitalière)']
    # df['Osmotherapy'] = df['Osmothérapie']
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

    # Drop rows with missing values in the feature to predict
    # df[df_mv[predict] != 0] = np.nan
    return df.dropna(axis=0, subset=[predict])


tasks_meta.append(TaskMeta(
    name='acid',
    db='TB',
    df_name='20000',
    predict='Acide tranexamique',
    keep=[
        'Trauma.center',
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
        'Pupil.anomaly.ph',
        'Pupil.anomaly.h',
        'Osmotherapy',
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
    rename=rename_acid,
    transform=transform_df_acid,
))
