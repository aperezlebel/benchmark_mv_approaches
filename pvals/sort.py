"""Sort pvals."""
import os
import pandas as pd


pvals_dirs = []
for db_dir in next(os.walk('.'))[1]:
    pvals_dirs += [f'{db_dir}/{d}' for d in next(os.walk(db_dir))[1]]

for pvals_dir in pvals_dirs:
    for filename in next(os.walk(pvals_dir))[2]:
        if 'pvals_filtered.csv' not in filename:
            continue  # not pvals, skipping

        pvals_path = f'{pvals_dir}/{filename}'
        pvals = pd.read_csv(pvals_path, header=None)

        pvals = pvals.sort_values(1, axis=0)

        basepath, ext = os.path.splitext(pvals_path)
        new_pvals_path = f'{basepath}_sorted{ext}'
        pvals.to_csv(new_pvals_path, index=None, header=None)
