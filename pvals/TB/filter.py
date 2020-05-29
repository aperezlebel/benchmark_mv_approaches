"""Remove geatures having some categories from pvals (e.g outcomes)."""
import pandas as pd


pvals_dirs = [
    'death/',
    'septic/',
]

categories = [
    'operation_outcomes',
]

features_in_category = dict()

for category in categories:
    with open(f'categories/{category}.txt', 'r') as file:
        features_in_category[category] = file.read().splitlines()


def remove_features_id(pvals, features_ids):

    for feature_id in features_ids:
        print(feature_id)
        pvals = pvals[~pvals[0].str.contains(feature_id)]

    return pvals



for pvals_dir in pvals_dirs:
    pvals = pd.read_csv(pvals_dir+'pvals.csv', header=None)

    # remove features

    for features_ids in features_in_category.values():
        pvals = remove_features_id(pvals, features_ids)

    pvals.to_csv(pvals_dir+'pvals_filtered.csv', index=None, header=None)
