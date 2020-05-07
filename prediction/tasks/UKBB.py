"""Build prediction tasks for the UKBB database."""
import copy

from .taskMeta import TaskMeta


tasks_meta = list()


# Task 1: Fluid intelligence prediction
def transform_df_fluid_intelligence(df, **kwargs):
    # Drop rows with missing values in the feature to predict
    predict = kwargs['meta'].predict

    return df.dropna(axis=0, subset=[predict])


tasks_meta.append(TaskMeta(
    name='fluid_intelligence',
    db='UKBB',
    df_name='40284',
    predict='20016-0.0',
    drop=[
        '20016-1.0',
        '20016-2.0'
    ],
    drop_contains=[
        '10136-',
        '10137-',
        '10138-',
        '10141-',
        '10144-',
        '10609-',
        '10610-',
        '10612-',
        '10721-',
        '10722-',
        '10740-',
        '10827-',
        '10860-',
        '10895-',  # pilots
        '20128-',
        '4935-',
        '4946-',
        '4957-',
        '4968-',
        '4979-',
        '4990-',
        '5001-',
        '5012-',
        '5556-',
        '5699-',
        '5779-',
        '5790-',
        '5866-',  # response
        '40001-',
        '40002-',
        '41202-',
        '41204',
        '20002-',  # large code
        '40006',
    ],
    transform=transform_df_fluid_intelligence,
    classif=False,
))

tasks_meta.append(TaskMeta(
    name='fluid_intelligence_light',
    db='UKBB',
    df_name='24440',
    predict='20016-0.0',
    drop=[
        '20016-1.0',
        '20016-2.0'
    ],
    drop_contains=[
        '10136-',
        '10137-',
        '10138-',
        '10141-',
        '10144-',
        '10609-',
        '10610-',
        '10612-',
        '10721-',
        '10722-',
        '10740-',
        '10827-',
        '10860-',
        '10895-',  # pilots
        '20128-',
        '4935-',
        '4946-',
        '4957-',
        '4968-',
        '4979-',
        '4990-',
        '5001-',
        '5012-',
        '5556-',
        '5699-',
        '5779-',
        '5790-',
        '5866-',  # response
        '40001-',
        '40002-',
        '41202-',
        '41204',
        '20002-',  # large code
        '40006',
    ],
    transform=transform_df_fluid_intelligence,
    classif=False,
))


tasks_meta.append(TaskMeta(
    name='breast',
    db='UKBB',
    df_name='40663',
    predict=None,
    keep=[
        '31-0.0',  # Sex
        '48-0.0',  # Waist circumpherence
        '2714-0.0',  # Age of menarche
        '3581-0.0',  # Age of menopause
        '2724-0.0',  # Had menopause
        '21001-0.0',  # BMI
        '23104-0.0',  # BMI
        '20116-0.0',  # Smoking status
        '1239-0.0',  # Current tobacco smoking
        '1249-0.0',  # Past tobacco smoking
        '22506-0.0',  # Tobacco smoking
        '201260-0.0',  # Ever smoked
        '1259-0.0',  # Smoking/smokers in household
        '3436-0.0',  # Age started smoking in current smokers
        '2867-0.0',  # Age started smoking in former smokers
        '20161-0.0',  # Pack years adult smoking as proportion of life span exposed to smoking
        '20162-0.0',  # Pack years of smoking
        '22611-0.0',  # Workplace had a lot of cigarette smoke from other people smoking
        '1787-0.0',  # Maternal smoking around birth
        '2794-0.0',  # Age started oral contraceptive pill
        '2794-0.0',  # Age started oral contraceptive pill
        '2804-0.0',  # Age when last used oral contraceptive pill
        '2784-0.0',  # Ever taken oral contraceptive pill
        '132122-0.0',  # Date N80 first reported (endometriosis)
        '41270-0.0',  # Diagnoses - ICD10
        '41271-0.0',  # Diagnoses - ICD9
        '41202-0.0',  # Diagnoses - main ICD10
        '41203-0.0',  # Diagnoses - main ICD9
        '41204-0.0',  # Diagnoses - secondary ICD10
        '41205-0.0',  # Diagnoses - secondary ICD9
        '41201-0.0',  # External causes - ICD10
        '40006-0.0',  # Type of cancer: ICD10
        '40013-0.0',  # Type of cancer: ICD9
    ],
    transform=None,
    classif=True,
))


tasks_meta.append(TaskMeta(
    name='breast_light',
    db='UKBB',
    df_name='40284',
    predict=None,
    keep=[
        '31-0.0',  # Sex
        '48-0.0',  # Waist circumpherence
    ],
    transform=None,
    classif=True,
))


