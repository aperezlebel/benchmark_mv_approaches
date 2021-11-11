import re
import os
from os.path import join
import argparse
import shutil
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=2500)
parser.add_argument('--root', type=str, default='results', dest='results_folder')
parser.add_argument('--export', type=str, default=None, dest='export_folder')
parser.add_argument('--allow-incomplete', type=bool, nargs='?', default=False, const=True, dest='allow_incomplete')

args = parser.parse_args()

filenames = [
    f'{args.n}_prediction.csv',
    f'{args.n}_probas.csv',
    f'{args.n}_times.csv',
]
rows = []


def file_len(f):
    for i, l in enumerate(f):
        pass
    return i + 1


print(f'Checking results for n={args.n}:')
for root, subdirs, files in os.walk(args.results_folder):
    counts = {}

    for filename in filenames:

        if filename in files:
            with open(join(root, filename)) as f:
                counts[filename] = file_len(f)

            # Extract trial and task from root
            print(root)
            res = re.search(join(args.results_folder, '/(.*)/RS'), root)
            if res is None:
                task = None
                trial = None
                counts[filename] = None
            else:
                task = res.group(1)
                res = re.search('RS0_T(.)_', root)
                trial = res.group(1)


            # print(task)
            # print(trial)
            # exit()

        else:
                counts[filename] = None


    if any(counts.values()):
        rows.append([root, task, trial]+list(counts.values()))
        if args.export_folder is not None:
            if args.allow_incomplete or counts[f'{args.n}_times.csv'] == 6:
                dest_folder = join(args.export_folder, Path(root).relative_to(args.results_folder))
                os.makedirs(dest_folder, exist_ok=True)
                for file in filenames+['strat_infos.yml', 'features.yml', 'task_infos.yml']:
                    if file in files:
                        shutil.copy(join(root, file), dest_folder)


df = pd.DataFrame(rows, columns=['path', 'task', 'trial']+filenames)

print(df)
df.to_csv('results_counts.csv')

df_incomplete = df.query(f'`{args.n}_times.csv` != 6')
df_incomplete.to_csv('results_incomplete.csv')
