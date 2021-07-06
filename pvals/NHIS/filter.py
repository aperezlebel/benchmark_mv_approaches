"""Remove features listed in exclude file."""
import os
import pandas as pd


def filter():
    pvals_dirs = next(os.walk('pvals/NHIS'))[1]

    exclude_filename = 'exclude.txt'

    def remove_features_id(pvals, features_ids):
        """Remove features from given pvals."""
        if not isinstance(features_ids, list):
            features_ids = [features_ids]

        for feature_id in features_ids:
            # Match exact feature or start with & followed by '_' (categorical)
            regex = f'(^{feature_id}$|^{feature_id}_)'
            pvals = pvals[~pvals[0].str.match(regex)]

        return pvals

    for pvals_dir in pvals_dirs:
        for filename in next(os.walk(f'pvals/NHIS/{pvals_dir}'))[2]:
            if 'pvals.csv' not in filename:
                continue  # not pvals, skipping

            pvals_path = f'pvals/NHIS/{pvals_dir}/{filename}'
            pvals = pd.read_csv(pvals_path, header=None, dtype=str)

            # remove features
            exclude_path = f'pvals/NHIS/{pvals_dir}/{exclude_filename}'
            with open(exclude_path, 'r') as file:
                for line in file:
                    pvals = remove_features_id(pvals, line.strip())

            basepath, ext = os.path.splitext(pvals_path)
            new_pvals_path = f'{basepath}_filtered{ext}'
            pvals.to_csv(new_pvals_path, index=None, header=None)
