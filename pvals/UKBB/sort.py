"""Sort pvals."""
import pandas as pd


pvals_dirs = [
    'breast_plain/',
    'parkinson_plain/',
    'melanomia_plain/',
]


for pvals_dir in pvals_dirs:
    pvals = pd.read_csv(pvals_dir+'pvals_filtered.csv', header=None)

    pvals = pvals.sort_values(1, axis=0)

    pvals.to_csv(pvals_dir+'pvals_filtered_sorted.csv', index=None, header=None)
