"""Filter featrues for UKBB from ukb40663.csv."""
import pandas as pd
import logging
import csv


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

dump_folder = 'extracted/'
sep = ','
encoding = 'ISO-8859-1'


def run_v2(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""

    # Features to keep
    df2 = pd.read_csv('extracted/ukb40663_features_filtered.csv')
    features_to_keep = set(df2['feature_name'])

    # All features
    df3 = pd.read_csv('UKBB/ukbb_tabular/csv/ukb40663.csv', nrows=0)
    features = set(df3.columns)

    # Features to drop
    features_to_drop = features - features_to_keep

    df = pd.read_csv('UKBB/ukbb_tabular/csv/ukb40663.csv', index_col='eid',
                     sep=sep, encoding=encoding)
    df.drop(features_to_drop, axis=1, inplace=True)

    df.to_csv(f'ukb40663_filtered.csv', quoting=csv.QUOTE_ALL)


def run_v1(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""

    # Features to keep
    df2 = pd.read_csv('extracted/ukb40663_features_filtered_v2.csv')

    features = set(df2['feature_name'])

    df = pd.read_csv('UKBB/ukbb_tabular/csv/ukb40663_filtered.csv', usecols=features,
                     index_col='eid', sep=sep, encoding=encoding)

    df_t = df.transpose()
    df_t.to_csv(f'extracted/ukb40663_filtered_v2_transposed.csv', quoting=csv.QUOTE_ALL)

    df.to_csv(f'extracted/ukb40663_filtered_v2.csv', quoting=csv.QUOTE_ALL)
