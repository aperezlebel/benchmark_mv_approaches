"""Remove features listed in exclude file."""
# ----------------
# USELESS FOR NOW
# ----------------
import os
import pandas as pd


pvals_dirs = next(os.walk('.'))[1]

exclude_filename = 'exclude.txt'


def remove_features_id(pvals, features_ids):
    """Remove features from given pvals."""
    if not isinstance(features_ids, list):
        features_ids = [features_ids]

    for feature_id in features_ids:
        # Match exact feature or start with and followed by '_' (categorical)
        regex = f'(^{feature_id}$|^{feature_id}_)'
        print(f'{feature_id} {regex}')
        pvals = pvals[~pvals[0].str.match(regex)]

    return pvals


for pvals_dir in pvals_dirs:
    for filename in next(os.walk(pvals_dir))[2]:
        if 'pvals.csv' not in filename:
            continue  # not pvals, skipping

        pvals_path = f'{pvals_dir}/{filename}'
        pvals = pd.read_csv(pvals_path, header=None, dtype=str)

        # remove features
        exclude_path = f'{pvals_dir}/{exclude_filename}'
        with open(exclude_path, 'r') as file:
            for line in file:
                pvals = remove_features_id(pvals, line.strip())

        basepath, ext = os.path.splitext(pvals_path)
        new_pvals_path = f'{basepath}_filtered{ext}'
        pvals.to_csv(new_pvals_path, index=None, header=None)
