import re
import os
from os.path import join

import pandas as pd

filenames = [
    '2500_prediction.csv',
    '2500_times.csv',
    '2500_probas.csv',
]
rows = []


def file_len(f):
    for i, l in enumerate(f):
        pass
    return i + 1


for root, subdirs, files in os.walk('results/'):
    counts = {}

    for filename in filenames:
        if filename in files:
            with open(join(root, filename)) as f:
                counts[filename] = file_len(f)

            # Extract trial and task from root
            print(root)
            res = re.search('results/(.*)/RS', root)
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

# print(rows)

df = pd.DataFrame(rows, columns=['path', 'task', 'trial']+filenames)

print(df)
df.to_csv('results_counts.csv')

df_incomplete = df.query('`2500_times.csv` != 6')
df_incomplete.to_csv('results_incomplete.csv')
