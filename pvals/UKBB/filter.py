"""Remove features having some categories from pvals (e.g outcomes)."""
import os
import pandas as pd


pvals_dirs = next(os.walk('.'))[1]

categories = [
    '713',
    '715',
    '718',
]

features_in_category = dict()

for category in categories:
    with open(f'categories/{category}.txt', 'r') as file:
        features_in_category[category] = file.read().splitlines()


def remove_features_id(pvals, features_ids):
    """Remove features from given pvals."""
    for feature_id in features_ids:
        print(feature_id)
        pvals = pvals[~pvals[0].str.contains(feature_id+'-')]

    return pvals


for pvals_dir in pvals_dirs:
    for filename in next(os.walk(pvals_dir))[2]:
        if 'pvals.csv' not in filename:
            continue  # not pvals, skipping

        pvals_path = f'{pvals_dir}/{filename}'
        pvals = pd.read_csv(pvals_path, header=None, dtype=str)

        # remove features
        for features_ids in features_in_category.values():
            pvals = remove_features_id(pvals, features_ids)

        basepath, ext = os.path.splitext(pvals_path)
        new_pvals_path = f'{basepath}_filtered{ext}'
        pvals.to_csv(new_pvals_path, index=None, header=None)
