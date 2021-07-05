"""Remove features having some categories from pvals (e.g outcomes)."""
import os
import pandas as pd
from tqdm import tqdm


def filter():
    pvals_dirs = next(os.walk('pvals/TB'))[1]

    categories = [
        'operation_outcomes',
    ]

    features_in_category = dict()

    for category in categories:
        with open(f'pvals/TB/categories/{category}.txt', 'r') as file:
            features_in_category[category] = file.read().splitlines()

    def remove_features_id(pvals, features_ids):
        """Remove features from given pvals."""
        for feature_id in features_ids:
            pvals = pvals[~pvals[0].str.contains(feature_id)]

        return pvals

    for pvals_dir in tqdm(pvals_dirs):
        for filename in next(os.walk(f'pvals/TB/{pvals_dir}'))[2]:
            if 'pvals.csv' not in filename:
                continue  # not pvals, skipping

            pvals_path = f'pvals/TB/{pvals_dir}/{filename}'
            pvals = pd.read_csv(pvals_path, header=None, dtype=str)

            # remove features
            for features_ids in features_in_category.values():
                pvals = remove_features_id(pvals, features_ids)

            basepath, ext = os.path.splitext(pvals_path)
            new_pvals_path = f'{basepath}_filtered{ext}'
            pvals.to_csv(new_pvals_path, index=None, header=None)
