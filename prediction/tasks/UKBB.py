"""Build prediction tasks for the UKBB database."""

import pandas as pd
import numpy as np

import database
from prediction import PredictionTask

df_name = '24440'
UKBB = database.UKBB()
# df_with_MV = UKBB.encoded_dataframes[df_name]
# df_mv = UKBB.encoded_missing_values[df_name]
# df_imputed = pd.read_csv('imputed/TB_20000_imputed_rounded_Iterative.csv',
#                          sep=';', index_col=0).astype(df_with_MV.dtypes)


# Task 1: Fluid intelligence prediction
to_predict_1 = '20016-0.0'
to_drop_1 = [
    '20016-1.0',
    '20016-2.0'
]
drop_contains_1 = [
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
]

def transform_df_1(df, to_predict):
    # Drop rows with missing values in the feature to predict
    return df.dropna(axis=0, subset=[to_predict])


fluid_with_MV = PredictionTask(
    db=UKBB,
    df_name=df_name,
    transform=lambda df: transform_df_1(df, to_predict_1),
    to_predict=to_predict_1,
    to_drop=to_drop_1,
    drop_contains=drop_contains_1
)

# All tasks
tasks = {
    'fluid_with_MV': fluid_with_MV
}
