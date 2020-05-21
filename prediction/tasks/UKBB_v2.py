"""Prediction tasks v2 for UKBB."""
import pandas as pd

from .task_v2 import TaskMeta
from .transform import Transform


task_metas = list()


# Task 1.1: Breast cancer prediction followng paper
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6558751/
# -----------------------------------------------------
# Define some features names that will be used in this task
ICD9 = '41271-0.0'
ICD9_main = '41203-0.0'
ICD9_sec = '41205-0.0'
ICD9_cancer = '40013-0.0'

ICD10 = '41270-0.0'
ICD10_main = '41202-0.0'
ICD10_sec = '41204-0.0'
ICD10_cancer = '40006-0.0'


# Define some helpers
def ICD9_equal(df, value):
    """Define a helper for ICD9 diagnsoses."""
    return ((df[ICD9] == value)
            | (df[ICD9_main] == value)
            | (df[ICD9_sec] == value))


def ICD10_equal(df, value):
    """Define a helper for ICD10 diagnsoses."""
    return ((df[ICD10] == value)
            | (df[ICD10_main] == value)
            | (df[ICD10_sec] == value))


def cancer_ICD10(df, value):
    """Define a helper for ICD10 cancer diagnsoses."""
    return df[ICD10_cancer] == value


def cancer_ICD9(df, value):
    """Define a helper for ICD9 cancer diagnsoses."""
    return df[ICD9_cancer] == value


# Define the callable used to select the indexes to keep in the df
def select_idx_breast(df):
    """Define the idx to keep from the database."""
    male_idx = df.index[df['31-0.0'] == 1]
    return df.drop(male_idx, axis=0)


breast_idx_transform = Transform(
    input_features=['31-0.0'],
    transform=select_idx_breast,
)


# Define the callable used to create the feature to predict
def define_predict_breast(df):
    """Callable used to define the feature to predict."""
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

    # Convert bool to {0, 1}
    df['C50'] = df['C50'].astype(int)

    return df


breast_predict_transform = Transform(
    input_features=[ICD9, ICD9_main, ICD9_sec, ICD9_cancer, ICD10, ICD10_main,
                    ICD10_sec, ICD10_cancer],
    transform=define_predict_breast,
    output_features=['C50'],
)


def define_new_features_breast(df):
    """Callable used to define new features from a bunch of features."""
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

    # Has cancer
    df['Cancer != C50'] = (
        (~df[ICD10_cancer].isna() | ~df[ICD9_cancer].isna()) & ~df['C50']
    )

    # Ovarian cancer
    df['C56'] = (
        cancer_ICD10(df, 'C56') | ICD10_equal(df, 'C56') |
        cancer_ICD9(df, '1830') | ICD9_equal(df, '1830')
    )

    # Convert bool to {0, 1}
    df['N80'] = df['N80'].astype(int)
    df['E28.2'] = df['E28.2'].astype(int)
    df['Cancer != C50'] = df['Cancer != C50'].astype(int)
    df['C56'] = df['C56'].astype(int)

    return df


breast_new_features_transform = Transform(
    input_features=[ICD9, ICD9_main, ICD9_sec, ICD9_cancer, ICD10, ICD10_main,
                    ICD10_sec, ICD10_cancer, 'C50'],
    transform=define_new_features_breast,
    output_features=['N80', 'E28.2', 'Cancer != C50', 'C56'],
)


# Define the features to keep
breast_keep_transform = Transform(
    input_features=[
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
        '20161-0.0',  # Pack years adult smoking as proportion of life span...
        '20162-0.0',  # Pack years of smoking
        '22611-0.0',  # Workplace had a lot of cigarette smoke...
        '1787-0.0',  # Maternal smoking around birth
        '2794-0.0',  # Age started oral contraceptive pill
        '2804-0.0',  # Age when last used oral contraceptive pill
        '2784-0.0',  # Ever taken oral contraceptive pill
    ]
)


task_metas.append(TaskMeta(
    name='breast_25',
    db='UKBB',
    df_name='40663_filtered',
    classif=True,
    idx_selection=breast_idx_transform,
    predict=breast_predict_transform,
    transform=breast_new_features_transform,
    select=breast_keep_transform,
    encode='all',
))


# Task 1.2: Breast cancer prediction using pvals
# ----------------------------------------------
# Define which features to keep
pvals = pd.read_csv('pvals/UKBB/breast_plain/pvals_filtered.csv', header=None,
                    index_col=0, squeeze=True)
pvals = pvals.sort_values()[:100]
breast_top_pvals = list(pvals.index)

breast_pvals_keep_transform = Transform(
    output_features=breast_top_pvals
)

task_metas.append(TaskMeta(
    name='breast_pvals',
    db='UKBB',
    df_name='40663_filtered',
    classif=True,
    idx_selection=breast_idx_transform,
    predict=breast_predict_transform,
    transform=None,
    select=breast_pvals_keep_transform,
    encode='all',
))


# Task 2.2: Melanomia prediction using pvals
# ------------------------------------------
# Define the callable used to create the feature to predict
def define_predict_skin(df):
    """Callable used to define the feature to predict."""
    # Melanoma and other malignant neoplasms of skin
    df['C43-C44'] = (
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
        cancer_ICD9(df, '1723') | ICD9_equal(df, '1723') |
        cancer_ICD9(df, '1725') | ICD9_equal(df, '1725') |
        cancer_ICD9(df, '1726') | ICD9_equal(df, '1726') |
        cancer_ICD9(df, '1727') | ICD9_equal(df, '1727') |
        cancer_ICD9(df, '1729') | ICD9_equal(df, '1729') |
        cancer_ICD9(df, '1730') | ICD9_equal(df, '1730') |
        cancer_ICD9(df, '1731') | ICD9_equal(df, '1731') |
        cancer_ICD9(df, '1732') | ICD9_equal(df, '1732') |
        cancer_ICD9(df, '1733') | ICD9_equal(df, '1733') |
        cancer_ICD9(df, '1734') | ICD9_equal(df, '1734') |
        cancer_ICD9(df, '1735') | ICD9_equal(df, '1735') |
        cancer_ICD9(df, '1736') | ICD9_equal(df, '1736') |
        cancer_ICD9(df, '1737') | ICD9_equal(df, '1737') |
        cancer_ICD9(df, '1739') | ICD9_equal(df, '1739')
    )

    # Convert bool to {0, 1}
    df['C43-C44'] = df['C43-C44'].astype(int)

    return df


skin_predict_transform = Transform(
    input_features=[ICD9, ICD9_main, ICD9_sec, ICD9_cancer, ICD10, ICD10_main,
                    ICD10_sec, ICD10_cancer],
    transform=define_predict_skin,
    output_features=['C43-C44'],
)

# Define which features to keep
pvals = pd.read_csv('pvals/UKBB/melanomia_plain/pvals_filtered.csv',
                    header=None, index_col=0, squeeze=True)
pvals = pvals.sort_values()[:100]
skin_top_pvals = list(pvals.index)

skin_pvals_keep_transform = Transform(
    output_features=skin_top_pvals
)

task_metas.append(TaskMeta(
    name='skin_pvals',
    db='UKBB',
    df_name='40663_filtered',
    classif=True,
    idx_selection=None,
    predict=skin_predict_transform,
    transform=None,
    select=skin_pvals_keep_transform,
    encode='all',
))