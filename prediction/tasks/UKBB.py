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

breast_keep = [
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
    '20160-0.0',  # Ever smoked
    '1259-0.0',  # Smoking/smokers in household
    '3436-0.0',  # Age started smoking in current smokers
    '2867-0.0',  # Age started smoking in former smokers
    '20161-0.0',  # Pack years adult smoking as proportion of life span exposed to smoking
    '20162-0.0',  # Pack years of smoking
    '22611-0.0',  # Workplace had a lot of cigarette smoke from other people smoking
    '1787-0.0',  # Maternal smoking around birth
    '2794-0.0',  # Age started oral contraceptive pill
    '2804-0.0',  # Age when last used oral contraceptive pill
    '2784-0.0',  # Ever taken oral contraceptive pill
    # '132122-0.0',  # Date N80 first reported (endometriosis)
    '41270-0.0',  # Diagnoses - ICD10
    '41271-0.0',  # Diagnoses - ICD9
    '41202-0.0',  # Diagnoses - main ICD10
    '41203-0.0',  # Diagnoses - main ICD9
    '41204-0.0',  # Diagnoses - secondary ICD10
    '41205-0.0',  # Diagnoses - secondary ICD9
    '41201-0.0',  # External causes - ICD10
    '40006-0.0',  # Type of cancer: ICD10
    '40013-0.0',  # Type of cancer: ICD9
]

# Used to extract features from the huge dataframe
tasks_meta.append(TaskMeta(
    name='breast_extract',
    db='UKBB',
    df_name='40663',
    predict=None,
    keep=breast_keep,
    transform=None,
    classif=True,
))


ICD9 = '41271-0.0'
ICD9_main = '41203-0.0'
ICD9_sec = '41205-0.0'

ICD10 = '41270-0.0'
ICD10_main = '41202-0.0'
ICD10_sec = '41204-0.0'


def ICD9_equal(df, value):
    return ((df[ICD9] == value)
            | (df[ICD9_main] == value)
            | (df[ICD9_sec] == value))


def ICD10_equal(df, value):
    return ((df[ICD10] == value)
            | (df[ICD10_main] == value)
            | (df[ICD10_sec] == value))


def cancer_ICD10(df, value):
    return df['40006-0.0'] == value


def cancer_ICD9(df, value):
    return df['40013-0.0'] == value


# Task 2: Breast cancer prediction
def transform_df_breast(df, **kwargs):
    # Keep only females
    male_idx = df.index[df['31-0.0'] == 1]
    df.drop(male_idx, axis=0, inplace=True)

    # People with endometriosis
    df['N80'] = (
        ICD10_equal(df, 'N800') | ICD9_equal(df, '6170') |
        ICD10_equal(df, 'N801') | ICD9_equal(df, '6171') |
        ICD10_equal(df, 'N802') | ICD9_equal(df, '6172') |
        ICD10_equal(df, 'N803') | ICD9_equal(df, '6173') |
        ICD10_equal(df, 'N804') | ICD9_equal(df, '6174') |
        ICD10_equal(df, 'N805') | ICD9_equal(df, '6175') |
        ICD10_equal(df, 'N806') | ICD9_equal(df, '6176') |
        ICD10_equal(df, 'N808') | ICD9_equal(df, '6178') |
        ICD10_equal(df, 'N809') | ICD9_equal(df, '6179')
    )

    # Polycystic ovarian syndrom
    df['E28.2'] = ICD10_equal(df, 'E282') | ICD9_equal(df, '2564')

    # Breast cancer
    df['C50'] = (
        cancer_ICD10(df, 'C500') | ICD10_equal(df, 'C500') |
        cancer_ICD10(df, 'C501') | ICD10_equal(df, 'C501') |
        cancer_ICD10(df, 'C502') | ICD10_equal(df, 'C502') |
        cancer_ICD10(df, 'C503') | ICD10_equal(df, 'C503') |
        cancer_ICD10(df, 'C504') | ICD10_equal(df, 'C504') |
        cancer_ICD10(df, 'C505') | ICD10_equal(df, 'C505') |
        cancer_ICD10(df, 'C506') | ICD10_equal(df, 'C506') |
        cancer_ICD10(df, 'C508') | ICD10_equal(df, 'C508') |
        cancer_ICD10(df, 'C509') | ICD10_equal(df, 'C509') |
        cancer_ICD9(df, '1740') | ICD9_equal(df, '1740') |
        cancer_ICD9(df, '1743') | ICD9_equal(df, '1743') |
        cancer_ICD9(df, '1744') | ICD9_equal(df, '1744') |
        cancer_ICD9(df, '1745') | ICD9_equal(df, '1745') |
        cancer_ICD9(df, '1748') | ICD9_equal(df, '1748') |
        cancer_ICD9(df, '1749') | ICD9_equal(df, '1749')
    )

    # Has cancer
    df['Cancer != C50'] = ~df['40006-0.0'].isna() & ~df['C50']

    # Ovarian cancer
    df['C56'] = (
        cancer_ICD10(df, 'C56') | ICD10_equal(df, 'C56') |
        cancer_ICD9(df, '1830') | ICD9_equal(df, '1830')
    )

    # Convert bool to {0, 1}
    df['N80'] = df['N80'].astype(int)
    df['E28.2'] = df['E28.2'].astype(int)
    df['C50'] = df['C50'].astype(int)
    df['Cancer != C50'] = df['Cancer != C50'].astype(int)
    df['C56'] = df['C56'].astype(int)

    return df.dropna(axis=0, subset=['C50'])


tasks_meta.append(TaskMeta(
    name='breast',
    db='UKBB',
    df_name='breast_40663',
    predict='C50',
    keep=breast_keep,
    keep_after_transform=[
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
        '20160-0.0',  # Ever smoked
        '1259-0.0',  # Smoking/smokers in household
        '3436-0.0',  # Age started smoking in current smokers
        '2867-0.0',  # Age started smoking in former smokers
        '20161-0.0',  # Pack years adult smoking as proportion of life span exposed to smoking
        '20162-0.0',  # Pack years of smoking
        '22611-0.0',  # Workplace had a lot of cigarette smoke from other people smoking
        '1787-0.0',  # Maternal smoking around birth
        '2794-0.0',  # Age started oral contraceptive pill
        '2804-0.0',  # Age when last used oral contraceptive pill
        '2784-0.0',  # Ever taken oral contraceptive pill
        'N80',
        'E28.2',
        'C50',
        'Cancer != C50',
        'C56',
    ],
    transform=transform_df_breast,
    classif=True,
))


def transform_df_breast_plain(df, **kwargs):
    # Keep only females
    male_idx = df.index[df['31-0.0'] == 1]
    df.drop(male_idx, axis=0, inplace=True)

    # Breast cancer
    df['C50'] = (
        cancer_ICD10(df, 'C500') | ICD10_equal(df, 'C500') |
        cancer_ICD10(df, 'C501') | ICD10_equal(df, 'C501') |
        cancer_ICD10(df, 'C502') | ICD10_equal(df, 'C502') |
        cancer_ICD10(df, 'C503') | ICD10_equal(df, 'C503') |
        cancer_ICD10(df, 'C504') | ICD10_equal(df, 'C504') |
        cancer_ICD10(df, 'C505') | ICD10_equal(df, 'C505') |
        cancer_ICD10(df, 'C506') | ICD10_equal(df, 'C506') |
        cancer_ICD10(df, 'C508') | ICD10_equal(df, 'C508') |
        cancer_ICD10(df, 'C509') | ICD10_equal(df, 'C509') |
        cancer_ICD9(df, '1740') | ICD9_equal(df, '1740') |
        cancer_ICD9(df, '1743') | ICD9_equal(df, '1743') |
        cancer_ICD9(df, '1744') | ICD9_equal(df, '1744') |
        cancer_ICD9(df, '1745') | ICD9_equal(df, '1745') |
        cancer_ICD9(df, '1748') | ICD9_equal(df, '1748') |
        cancer_ICD9(df, '1749') | ICD9_equal(df, '1749')
    )

    df['C50'] = df['C50'].astype(int)

    return df.dropna(axis=0, subset=['C50'])


tasks_meta.append(TaskMeta(
    name='breast_plain',
    db='UKBB',
    df_name='40663',
    predict='C50',
    transform=transform_df_breast_plain,
    classif=True,
))


def transform_df_melanomia_plain(df, **kwargs):
    # Malignant melanoma and neoplasms of skin
    df['C43-44'] = (
        cancer_ICD10(df, 'C430') | ICD10_equal(df, 'C430') |
        cancer_ICD10(df, 'C431') | ICD10_equal(df, 'C431') |
        cancer_ICD10(df, 'C432') | ICD10_equal(df, 'C432') |
        cancer_ICD10(df, 'C433') | ICD10_equal(df, 'C433') |
        cancer_ICD10(df, 'C434') | ICD10_equal(df, 'C434') |
        cancer_ICD10(df, 'C435') | ICD10_equal(df, 'C435') |
        cancer_ICD10(df, 'C436') | ICD10_equal(df, 'C436') |
        cancer_ICD10(df, 'C437') | ICD10_equal(df, 'C437') |
        cancer_ICD10(df, 'C438') | ICD10_equal(df, 'C438') |
        cancer_ICD10(df, 'C439') | ICD10_equal(df, 'C439') |
        cancer_ICD10(df, 'C440') | ICD10_equal(df, 'C440') |
        cancer_ICD10(df, 'C441') | ICD10_equal(df, 'C441') |
        cancer_ICD10(df, 'C442') | ICD10_equal(df, 'C442') |
        cancer_ICD10(df, 'C443') | ICD10_equal(df, 'C443') |
        cancer_ICD10(df, 'C444') | ICD10_equal(df, 'C444') |
        cancer_ICD10(df, 'C445') | ICD10_equal(df, 'C445') |
        cancer_ICD10(df, 'C446') | ICD10_equal(df, 'C446') |
        cancer_ICD10(df, 'C447') | ICD10_equal(df, 'C447') |
        cancer_ICD10(df, 'C448') | ICD10_equal(df, 'C448') |
        cancer_ICD10(df, 'C449') | ICD10_equal(df, 'C449') |
        cancer_ICD9(df, '1720') | ICD9_equal(df, '1720') |
        cancer_ICD9(df, '1722') | ICD9_equal(df, '1722') |
        cancer_ICD9(df, '1723') | ICD9_equal(df, '1723') |
        cancer_ICD9(df, '1724') | ICD9_equal(df, '1724') |
        cancer_ICD9(df, '1725') | ICD9_equal(df, '1725') |
        cancer_ICD9(df, '1726') | ICD9_equal(df, '1726') |
        cancer_ICD9(df, '1727') | ICD9_equal(df, '1727') |
        cancer_ICD9(df, '1728') | ICD9_equal(df, '1728') |
        cancer_ICD9(df, '1729') | ICD9_equal(df, '1729') |
        cancer_ICD9(df, '1730') | ICD9_equal(df, '1730') |
        cancer_ICD9(df, '1732') | ICD9_equal(df, '1732') |
        cancer_ICD9(df, '1733') | ICD9_equal(df, '1733') |
        cancer_ICD9(df, '1734') | ICD9_equal(df, '1734') |
        cancer_ICD9(df, '1735') | ICD9_equal(df, '1735') |
        cancer_ICD9(df, '1736') | ICD9_equal(df, '1736') |
        cancer_ICD9(df, '1737') | ICD9_equal(df, '1737') |
        cancer_ICD9(df, '1738') | ICD9_equal(df, '1738') |
        cancer_ICD9(df, '1739') | ICD9_equal(df, '1739')
    )

    df['C43-44'] = df['C43-44'].astype(int)

    return df.dropna(axis=0, subset=['C43-44'])


tasks_meta.append(TaskMeta(
    name='melanomia_plain',
    db='UKBB',
    df_name='40663',
    predict='C43-44',
    transform=transform_df_melanomia_plain,
    classif=True,
))
