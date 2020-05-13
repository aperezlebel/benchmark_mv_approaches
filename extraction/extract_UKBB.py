"""Filter featrues for UKBB from ukb40663.csv."""
import pandas as pd
import logging
import csv


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

dump_folder = 'extracted/'


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""

    # Features to keep
    df2 = pd.read_csv('extracted/ukb40663_features_filtered.csv')

    features = set(df2['feature_name'])
    print(features)

    df = pd.read_csv('UKBB/ukbb_tabular/csv/ukb40663.csv', usecols=features,
                     index_col='eid')

    df.to_csv(f'ukb40663_filtered.csv', quoting=csv.QUOTE_ALL)
